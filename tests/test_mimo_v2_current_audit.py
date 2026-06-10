from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def test_mimo_latest_speed_uses_current_text_runtime_boundary() -> None:
    evidence = audit._latest_decode_speed_evidence(
        {
            "status": "open",
            "speed_boundary": {
                "accepted_prior_speed_artifact": (
                    "build/current-all-local-model-smoke-mimo-v25-jangtq2-live-runtime-proof-20260609/"
                    "JANGQ_MiMo-V2.5-JANGTQ_2/result.json"
                ),
                "accepted_prior_generation_tps": 51.445792193622715,
                "current_source_long_decode_artifact": (
                    "build/current-mimo-v25-jangtq2-current-source-long-decode-speed-after-capability-fix-20260609/"
                    "result.json"
                ),
                "current_source_wall_tps": 36.6158,
                "current_source_server_tps": 36.9,
                "status": (
                    "open_because_current_source_long_decode_is_below_40_even_though_prior_live_smoke_crossed_50"
                ),
            },
        }
    )

    assert evidence["classification"] == "current_text_runtime_release_path_speed_boundary"
    assert evidence["speed_blocked"] is True
    assert evidence["bundle_decode_tps"] == 36.9
    assert evidence["stale_forced_mllm_generation_tps"] == 51.445792193622715


def test_mimo_bookend_quantization_audit_accepts_q8_sidecars(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "MiMo-V2.5-JANGTQ_2"
    model_path.mkdir()
    _write_json(
        model_path / "model.safetensors.index.json",
        {
            "weight_map": {
                "lm_head.weight": "bookends.safetensors",
                "lm_head.scales": "bookends.safetensors",
                "lm_head.biases": "bookends.safetensors",
                "model.embed_tokens.weight": "bookends.safetensors",
                "model.embed_tokens.scales": "bookends.safetensors",
                "model.embed_tokens.biases": "bookends.safetensors",
            }
        },
    )
    (model_path / "bookends.safetensors").write_bytes(b"stub")

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    class FakeSafeOpen:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_tensor(self, key):
            if key.endswith(".weight"):
                return FakeTensor((152576, 1024), "uint32")
            return FakeTensor((152576, 64), "float16")

    monkeypatch.setitem(
        sys.modules,
        "safetensors",
        types.SimpleNamespace(safe_open=FakeSafeOpen),
    )

    result = audit._bookend_quantization_audit(model_path)

    assert result["status"] == "pass"
    assert result["missing_keys"] == []
    assert result["q8_shape_compatible"] is True
    assert result["keys"]["lm_head.weight"]["dtype"] == "uint32"
    assert "does not clear MiMo exactness" in result["boundary"]


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
                "class MiMoVisionPatchEmbed",
                "def embed_patches(self, pixel_values):",
                "self.patch_embed = MiMoVisionPatchEmbed(",
                "flat_patch_width",
                "class MiMoVisionPatchMerger",
                "def merge_patches(self, hidden_states):",
                "self.merger = MiMoVisionPatchMerger(",
                "self.mlp = [",
                "nn.Linear(self.hidden_size, dim)",
                "class MiMoVisionAttention",
                "class MiMoVisionBlock",
                "class MiMoVisionSwiGLUMLP",
                "def run_blocks(self, hidden_states, grid_thw=None):",
                "mx.fast.scaled_dot_product_attention",
                "self.blocks = [",
                "def rot_pos_emb(self, grid_thw):",
                "def get_window_index_1d(self, grid_thw, col: bool = True):",
                "reverse_window_index_1d_col = mx.argsort(window_index_1d_col)",
                "position_embeddings=position_embeddings",
                "x = self.run_blocks(x, grid_thw=grid_thw)",
                "return self.merge_patches(x)",
                "self.visual = (",
                "if self.visual is None:",
                "image_embeds = self.visual(",
                "video_embeds = self.visual(",
                "image_grid_thw=image_grid_thw",
                "video_grid_thw=video_grid_thw",
                "class AudioProjection",
                "class AudioModel(nn.Module):",
                "self.audio_encoder = (",
                "def project_local_audio(self, audio_hidden):",
                "audio_embeds = self.audio_encoder(audio_embeds=audio_arr)",
                "class MiMoAudioLocalTransformer",
                "class MiMoAudioLocalAttention",
                "self.input_local_transformer = MiMoAudioLocalTransformer",
                "def process_audio_codes(self, audio_codes, speech_embeddings):",
                "speech_embeddings=self.speech_embeddings",
                "_pad_and_group_mimo_v2_audio_codes",
                "class MiMoAudioEuclideanCodebook",
                "class MiMoAudioResidualVectorQuantizer",
                "def encode(self, hidden_states",
                "def decode(self, codes",
                "mx.argmax(dist, axis=-1)",
                "def load_mimo_audio_rvq_from_bundle",
                "audio_tokenizer/model.safetensors",
                "encoder.quantizer.vq.layers",
                "MiMoAudioResidualVectorQuantizer(codebooks",
                "class MiMoAudioTokenizerConfig",
                "def get_output_length(self, mel_len",
                "def get_code_length(self, mel_len",
                "def plan_mimo_audio_mel_segments",
                "segments_per_mel",
                "code_lengths",
                "class MiMoAudioTokenizer",
                "class MiMoAudioTokenizerEncoder",
                "def encode_audio_to_codes",
                "def load_mimo_audio_tokenizer_from_bundle",
                "num_quantizers",
                "codebook_size",
                "n_mels",
                "return_codes_only",
                "def _apply_mimo_v2_media_weights(self):",
                "_mimo_v2_assign_weight(self, key, value)",
                "MiMo-V2 load assigned %d preserved media tensors",
                "MiMo-V2.5 JANG_2L vision input is not wired",
                'key.startswith("model.mtp.")',
            ]
        )
    )
    (tmp_path / "vmlx_engine/mllm_batch_generator.py").write_text(
        "\n".join(
            [
                "def _mllm_media_cache_extra_keys(request: Any)",
                "audio_codes",
                "audio_embeds",
                "audio_features",
                "cache_extra_keys=_cache_extra_keys",
                "cache_extra_keys=_ssm_extra_keys",
                'kwargs["audio_codes"] = request.audio_codes',
                'kwargs["audio_embeds"] = request.audio_embeds',
                'kwargs["audio_embeds"] = request.audio_features',
                "has_media_payload = has_images or has_audio_payload",
                "not has_media_payload",
                'request.extra_kwargs.pop("audio_codes", None)',
                'request.extra_kwargs.pop("audio_embeds", None)',
                'request.extra_kwargs.pop("audio_features", None)',
                "request.audio_codes = _ensure_mx_array(",
                "request.audio_embeds = _ensure_mx_array(",
                "request.audio_features = _ensure_mx_array(",
                "audio: Optional[List",
                "process_audio_input",
                "not request.images and not request.videos and not request.audio",
                "audio=all_audio",
                'kwargs["audio"] = audio',
                "skip_audios_alias",
            ]
        )
    )
    (tmp_path / "vmlx_engine/mllm_scheduler.py").write_text(
        "\n".join(
            [
                "audio: Optional[List[Any]] = None",
                "audio=request.audio",
                "audio=audio",
                "not images and not videos and not audio",
            ]
        )
    )
    (tmp_path / "vmlx_engine/engine").mkdir(parents=True)
    (tmp_path / "vmlx_engine/engine/batched.py").write_text(
        "\n".join(
            [
                "def _extract_audio_content",
                "extracted_audio = self._extract_audio_content(messages)",
                "all_audio = (audio or []) + extracted_audio",
                "audio=all_audio if all_audio else None",
            ]
        )
    )
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/test_zaya_runtime.py").write_text(
        "\n".join(
            [
                "test_mllm_media_cache_key_includes_audio_codes",
                "test_mllm_audio_payloads_are_media_cache_context",
            ]
        )
    )
    (tmp_path / "tests/test_mllm_scheduler_cache.py").write_text(
        "\n".join(
            [
                "test_mllm_audio_payload_prefill_uses_model_wrapper_not_text_fast_path",
                "test_mllm_processor_audio_outputs_are_promoted_to_request_fields",
                "test_mllm_processor_direct_forwards_raw_audio_to_processor",
                "test_mllm_processor_direct_omits_invalid_audios_alias_for_mimo_v2_processor",
                "test_mllm_scheduler_and_batched_engine_route_raw_audio_requests",
            ]
        )
    )
    (tmp_path / "tests/test_mimo_v2_media_runtime.py").write_text(
        "test_mimo_v2_audio_residual_vector_quantizer_matches_nearest_codebook\n"
        "test_mimo_v2_audio_rvq_loader_reads_bundle_codebooks\n"
        "test_mimo_v2_audio_tokenizer_config_and_mel_segment_plan\n"
        "test_mimo_v2_audio_tokenizer_executes_mel_to_audio_codes\n"
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
        tmp_path / "build/current-mimo-v2-switchglu-selected-expert-parity-20260609.json",
        {
            "got_shape": [2, 3, 4096],
            "manual_shape": [2, 3, 4096],
            "max_abs_diff": 0.0007,
            "mean_abs_diff": 0.00009,
        },
    )
    _write_json(
        tmp_path / audit.CACHE_VS_NOCACHE_ARTIFACT,
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
        / "build/current-all-local-model-smoke-mimo-v25-jangtq2-live-runtime-proof-20260609/JANGQ_MiMo-V2.5-JANGTQ_2/result.json",
        {
            "status": "fail",
            "results": [
                {
                    "name": "MiMo-V2.5-JANGTQ_2",
                    "relative_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
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
                    ]
                    + [
                        {
                            "label": "vl_blue_image",
                            "code": 200,
                            "content": "Blue",
                            "validation_failures": [],
                        },
                        {
                            "label": "vl_blue_image_repeat",
                            "code": 200,
                            "content": "Blue",
                            "validation_failures": [],
                        },
                        {
                            "label": "vl_red_image_changed",
                            "code": 200,
                            "content": "Red",
                            "validation_failures": [],
                        },
                        {
                            "label": "text_no_media_after_image",
                            "code": 200,
                            "content": "NONE",
                            "validation_failures": [],
                        },
                        {
                            "label": "vl_blue_video",
                            "code": 200,
                            "content": "Blue",
                            "validation_failures": [],
                        },
                        {
                            "label": "text_no_media_after_video",
                            "code": 200,
                            "content": "NONE",
                            "validation_failures": [],
                        },
                    ],
                    "l2_restart": {
                        "summary": {
                            "status": "pass",
                            "pass": True,
                            "reason": "pass",
                            "cache_hit_tokens": 67,
                            "disk_hits": 2,
                            "cache_summary": {
                                "cache_hit_tokens": 67,
                                "disk_hits": 2,
                                "has_cache_hit": True,
                            },
                        }
                    },
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
        tmp_path
        / "build/current-all-local-model-smoke-mimo-v25-jang2l-live-refresh-20260608/summary.json",
        {
            "status": "fail",
            "results": [
                {
                    "name": "MiMo-V2.5-JANG_2L",
                    "relative_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L",
                    "row": {
                        "capabilities": {
                            "modalities": ["text"],
                            "preserved_modalities": ["vision", "audio"],
                            "unwired_modalities": ["vision", "audio"],
                        }
                    },
                    "status": "probe_failed",
                    "capabilities": {
                        "body": {
                            "modalities": ["text"],
                            "media": {
                                "runtime_modalities": ["text"],
                                "declared_modalities": [
                                    "text",
                                    "vision",
                                    "image",
                                    "video",
                                    "audio",
                                ],
                                "preserved_modalities": [
                                    "vision",
                                    "image",
                                    "video",
                                    "audio",
                                ],
                                "unwired_modalities": [
                                    "vision",
                                    "image",
                                    "video",
                                    "audio",
                                ],
                                "status_by_modality": {
                                    "text": "runtime_supported",
                                    "vision": "memory_gated",
                                    "image": "memory_gated",
                                    "video": "memory_gated",
                                    "audio": "memory_gated",
                                },
                                "memory_gate": {
                                    "applicable": True,
                                    "safe": False,
                                    "reason": "insufficient_metal_working_set_headroom",
                                },
                            },
                        }
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "code": 200,
                            "content": "ACK",
                            "validation_failures": [],
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "code": 200,
                            "content": "ACK",
                            "usage": {
                                "prompt_tokens_details": {
                                    "cached_tokens": 60,
                                    "cache_detail": "paged",
                                }
                            },
                            "validation_failures": [],
                        },
                        {
                            "label": "mimo_structured_json_sentinel",
                            "code": 200,
                            "content": "{\"status\":\"ok\",\"value\":\"B7-CAT-09\",\"count\":1}",
                            "validation_failures": [
                                {
                                    "label": "mimo_structured_json_sentinel",
                                    "reason": "json_exact_object_mismatch",
                                    "expected": {
                                        "status": "ok",
                                        "value": "B7-CAT-09",
                                        "count": 3,
                                    },
                                    "actual": {
                                        "status": "ok",
                                        "value": "B7-CAT-09",
                                        "count": 1,
                                    },
                                }
                            ],
                        }
                    ],
                    "failures": [
                        {
                            "label": "mimo_structured_json_sentinel",
                            "reason": "json_exact_object_mismatch",
                        }
                    ],
                    "l2_restart": {
                        "label": "text_cache_repeat_l2_restart",
                        "status": "completed",
                        "code": 200,
                        "content": "ACK",
                        "usage": {
                            "prompt_tokens_details": {
                                "cached_tokens": 60,
                                "cache_detail": "paged+disk",
                            }
                        },
                        "cache_stats": {
                            "scheduler_stats": {
                                "cache_hit_tokens": 60,
                                "cache_hit_tokens_by_detail": {"paged+disk": 60},
                            },
                            "block_disk_cache": {"disk_hits": 1},
                        },
                        "summary": {
                            "status": "pass",
                            "pass": True,
                            "reason": "pass",
                            "disk_hits": 1,
                            "cache_hit_tokens": 60,
                        },
                    },
                    "server_log_tail": "MiMo-V2 text prefill using chunked path under tight memory. Application shutdown complete.",
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
        tmp_path / "build/current-mimo-v2-mllm-inputs-embeds-interface-proof-20260609.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path
        / "build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json",
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
        tmp_path / audit.TEXT_ROUTE_ARTIFACT,
        {
            "status": "open",
            "fix_status": "partial",
            "live_rows": {
                "long_prompt_first_request": {
                    "sentinel_recall_pass": True,
                    "oom_cleared": True,
                },
                "cache_repeat_1": {
                    "empty_visible_cleared": True,
                    "exact_expected_ack_cache_742": False,
                },
                "cache_repeat_2": {
                    "empty_visible_cleared": True,
                    "exact_expected_ack_cache_742": False,
                },
                "tool_required": {"tool_call_pass": True},
            },
            "remaining_blockers": [
                "decode_speed_below_release_target",
                "exact_cache_prompt_following_still_returns_ACK_not_ACK_CACHE_742",
                "prefix_paged_l2_cache_hit_not_reproved_on_continuous_batching_route",
                "source_vs_quant_first_divergence_missing",
                "mimo_vl_audio_video_unwired",
            ],
        },
    )
    _write_json(
        tmp_path
        / "build/current-mimo-v2-jangtq2-cb-cache-lossless-auto-live-20260609.json",
        {
            "status": "pass",
            "summary": {
                "memory_pressure_cleared": True,
                "exact_repeat_1": False,
                "exact_repeat_2": False,
                "no_think_tags": True,
                "cache_hit_tokens": 46,
                "l2_tokens_on_disk": 78,
                "native_cache": {
                    "cache_type": "mixed_swa_kv",
                    "storage_quantization": {"enabled": False, "bits": None},
                },
                "texts": {
                    "exact_repeat_1": "ACKCB-742",
                    "exact_repeat_2": "ACKCB-742",
                    "first_token_system_probe": "OK.",
                },
                "finish_reasons": {
                    "exact_repeat_1": "stop",
                    "exact_repeat_2": "stop",
                    "first_token_system_probe": "stop",
                },
            },
            "rows": [
                {
                    "name": "exact_repeat_1",
                    "ok": True,
                    "status": 200,
                    "elapsed_s": 1.9,
                    "usage": {"completion_tokens": 7},
                    "text": "ACKCB-742",
                },
                {
                    "name": "exact_repeat_2",
                    "ok": True,
                    "status": 200,
                    "elapsed_s": 0.24,
                    "usage": {"completion_tokens": 7},
                    "text": "ACKCB-742",
                },
                {
                    "name": "first_token_system_probe",
                    "ok": True,
                    "status": 200,
                    "error": None,
                    "text": "OK.",
                },
            ],
            "log_tail": [],
        },
    )
    _write_json(
        tmp_path / audit.NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
        {
            "status": "open",
            "classification": "model_generated_literal_mutation_after_valid_parser_structure",
            "source_vs_quant_load_performed": False,
            "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
            "excluded_surfaces": {
                "parser_argument_rewrite": True,
                "prefix_paged_l2_or_kv_quant_primary_cause": True,
                "hidden_stochastic_sampling_primary_cause": True,
            },
            "unresolved_surfaces": {
                "artifact_quantization_or_decode_logits_quality": True,
                "source_vs_quant_first_divergence": True,
            },
        },
    )
    _write_json(
        tmp_path / audit.JANGTQ2_SOURCE_MEDIA_PROOF_ARTIFACT,
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
                "fresh_process_l2_restore": True,
                "release_clearance": True,
            },
        },
    )
    _write_json(
        tmp_path / audit.JANGTQ2_DEV_APP_VIDEO_MEDIA_PROOF_ARTIFACT,
        {
            "status": "open",
            "transport_status": "pass",
            "semantic_status": "fail",
            "checks": {
                "real_electron_dev_build": True,
                "real_model_loaded": True,
                "mllm_flag_used": True,
                "server_media_diag_video_url": True,
                "http_200": True,
                "block_l2_write": True,
                "video_semantic_red_fixture": False,
            },
        },
    )
    _write_json(
        tmp_path / audit.JANGTQ2_VIDEO_CACHE_PROOF_ARTIFACT,
        {
            "status": "open",
            "proven": {
                "real_mimo_jangtq2_model_loaded": True,
                "video_request_reaches_runtime": True,
                "repeated_video_request_http_200": True,
                "video_pixel_cache_hit_after_first_request": True,
                "video_cache_contract_preserves_video_tensor_fields_in_source": True,
            },
        },
    )

    result = audit.build_audit(tmp_path, model_path, manifest)

    assert result["status"] == "open"
    assert result["local_release_clearance"] is False
    assert result["component_ok"]["manifest_integrity"] is True
    assert result["component_ok"]["text_cache_narrow"] is True
    assert result["component_ok"]["cache_vs_nocache_next_token"] is True
    assert result["component_ok"]["long_prompt_coherence"] is True
    assert result["component_ok"]["tool_protocol"] is True
    assert result["component_ok"]["exact_cache_prompt_following"] is True
    assert result["component_ok"]["decode_speed_target"] is False
    assert result["component_ok"]["system_prompt_first_token_stop"] is True
    assert result["component_ok"]["cb_system_prompt_working_set_pressure"] is True
    assert result["component_ok"]["source_vs_quant_first_divergence"] is False
    assert result["component_ok"]["source_vs_quant_policy_skipped"] is True
    assert result["component_ok"]["source_vs_quant_requirement_satisfied"] is True
    assert result["component_ok"]["text_route_long_prompt_supersedes_length_sweep"] is True
    assert result["component_ok"]["api_cache_responses_contract"] is True
    assert result["component_ok"]["media_weights_preserved"] is False
    assert result["component_ok"]["media_runtime_capabilities_safe"] is False
    assert result["component_ok"]["media_model_metadata_text_only_contract"] is False
    assert result["component_ok"]["media_runtime_implementation"] is True
    assert result["component_ok"]["mimo_media_wired"] is False
    assert result["component_ok"]["manual_sink_does_not_clear_length_generation"] is True
    assert result["component_ok"]["disable_sink_does_not_clear_length_generation"] is True
    assert (
        result["diagnostics"]["mimo_media_runtime"]["classification"]
        == "forced_mllm_media_transport_and_cache_live_semantics_red"
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"][
            "runtime_media_wired_superseded_by_route_proof"
        ]
        is True
    )
    assert (
        result["diagnostics"]["mimo_jangtq2_media_route"]["transport_proven"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_jangtq2_media_route"]["semantic_quality_proven"]
        is False
    )
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
    assert result["diagnostics"]["mimo_media_runtime"]["vision_patch_embed_module"] is True
    assert result["diagnostics"]["mimo_media_runtime"]["vision_merger_module"] is True
    assert (
        result["diagnostics"]["mimo_media_runtime"]["vision_attention_block_modules"]
        is True
    )
    assert result["diagnostics"]["mimo_media_runtime"]["vision_grid_forward"] is True
    assert result["diagnostics"]["mimo_media_runtime"]["model_vision_bridge"] is True
    assert result["diagnostics"]["mimo_media_runtime"]["audio_projection_bridge"] is True
    assert (
        result["diagnostics"]["mimo_media_runtime"]["audio_code_local_transformer"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["audio_tokenizer_rvq_codebook"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"][
            "audio_tokenizer_rvq_weight_loader"
        ]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"][
            "audio_tokenizer_config_segmenter"
        ]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["audio_tokenizer_model_execution"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["local_mimo_v2_multimodal_module"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["source_media_components_present"]
        is True
    )
    assert result["diagnostics"]["mimo_media_runtime"]["runtime_media_wired"] is False
    assert (
        result["diagnostics"]["mimo_media_runtime"]["missing_mimo_v2_multimodal_module"]
        is False
    )
    assert result["diagnostics"]["mimo_media_runtime"]["media_weight_assignment"] is True
    assert (
        result["diagnostics"]["mimo_media_runtime"]["media_aware_prefix_l2_cache"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["audio_payload_forwarding"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["audio_processor_output_bridge"]
        is True
    )
    assert (
        result["diagnostics"]["mimo_media_runtime"]["raw_audio_request_ingestion"]
        is True
    )
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
    assert (
        "visual patch embedding module"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "vision merger to 4096 hidden size"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "vision qkv attention and MLP block modules"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "grid-aware rotary/window index reorder and end-to-end ViT forward"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "model-level image/video request bridge to MiMo vision tower"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio tokenizer model execution bridge (waveform/mel -> 20-channel audio_codes)"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio projection bridge to text hidden size"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio code local transformer forward"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio tokenizer RVQ codebook weight loader"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio tokenizer config and mel segment planner"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "media weight assignment to runtime modules"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "media-aware prefix/L2 cache proof"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "audio payload forwarding through MLLM wrapper"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "processor-produced audio tensor field bridge"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert (
        "raw audio request bridge from API/UI to processor"
        not in result["diagnostics"]["mimo_media_runtime"]["missing_runtime_components"]
    )
    assert result["diagnostics"]["prompt_shape_first_token"]["blocked"] is True
    assert (
        result["diagnostics"]["prompt_shape_first_token"]["classification"]
        == "decode_loop_or_model_artifact_first_token_stop"
    )
    assert result["diagnostics"]["prompt_shape_first_token"]["failing_top_token"][
        "text"
    ] == "<|im_end|>"
    assert result["diagnostics"]["cb_metal_baseline"]["memory_pressure_cleared"] is True
    assert (
        result["diagnostics"]["cb_metal_baseline"]["storage_quantization_disabled"]
        is True
    )
    assert result["diagnostics"]["cb_metal_baseline"]["cache_repeat_stable"] is True
    assert result["diagnostics"]["cb_metal_baseline"]["cache_repeat_exact"] is False
    assert result["diagnostics"]["cb_metal_baseline"]["cache_hit_tokens"] == 46
    assert result["diagnostics"]["cb_metal_baseline"]["l2_tokens_on_disk"] == 78
    assert result["diagnostics"]["cb_metal_baseline"]["native_cache_type"] == "mixed_swa_kv"
    assert result["component_ok"]["mimo_jangtq2_live_media_l2"] is False
    assert result["component_ok"]["mimo_jang2l_live_media_l2"] is False
    assert result["component_ok"]["mimo_jang2l_l2_restart_cache_hit"] is True
    assert result["component_ok"]["mimo_jang2l_l2_restart_visible_output"] is True
    assert (
        result["component_ok"]["mimo_jang2l_media_capability_downscoped_to_text"]
        is False
    )
    assert result["component_ok"]["mimo_jang2l_media_capability_memory_gated"] is False
    assert result["component_ok"]["mimo_jang2l_tool_long_prompt_metal_oom"] is True
    assert result["component_ok"]["mimo_jang2l_tight_memory_prompt_budget"] is True
    assert result["component_ok"]["mimo_jang2l_media_prefill_budget"] is True
    assert (
        result["component_ok"]["mimo_jang2l_post_media_working_set_pressure"] is True
    )
    assert "mimo_jangtq2_live_media_l2_missing" in result["blockers"]
    assert "mimo_jang2l_live_media_l2_missing" in result["blockers"]
    assert "mimo_media_runtime_implementation_missing" not in result["blockers"]
    assert "mimo_exact_cache_prompt_following_blocked" not in result["blockers"]
    assert "mimo_jang2l_media_capability_downscoped_to_text" not in result["blockers"]
    assert "mimo_jang2l_media_capability_memory_gated" in result["blockers"]
    assert "mimo_jang2l_l2_restart_visible_output_blocked" not in result["blockers"]
    assert "mimo_jang2l_tool_long_prompt_metal_oom_blocked" not in result["blockers"]
    assert "mimo_jang2l_tight_memory_prompt_budget_blocked" not in result["blockers"]
    assert "mimo_jang2l_media_prefill_budget_blocked" not in result["blockers"]
    assert (
        "mimo_jang2l_post_media_working_set_pressure_blocked"
        not in result["blockers"]
    )
    assert (
        result["diagnostics"]["jang2l_all_local_smoke"]["bundle_kind"] == "jang2l"
    )
    assert (
        result["diagnostics"]["jang2l_all_local_smoke"]["artifact_exactness_boundary"][
            "classification"
        ]
        == "model_generated_literal_mutation_after_valid_parser_structure"
    )
    assert result["diagnostics"]["jang2l_all_local_smoke"]["metal_oom_crash"] is False
    assert (
        result["diagnostics"]["jang2l_all_local_smoke"][
            "typed_tight_memory_prompt_rejection"
        ]
        is False
    )
    assert (
        result["diagnostics"]["jang2l_all_local_smoke"][
            "typed_media_prefill_budget_rejection_labels"
        ]
        == []
    )
    assert (
        result["diagnostics"]["jang2l_all_local_smoke"][
            "post_media_working_set_pressure_rejection_labels"
        ]
        == []
    )
    assert result["diagnostics"]["cb_metal_baseline"]["memory_pressure_cleared"] is True
    assert result["diagnostics"]["api_cache_responses_contract"]["pass"] is True
    assert result["diagnostics"]["source_vs_quant_policy_skip"]["policy_skipped"] is True
    assert (
        result["diagnostics"]["source_vs_quant_policy_skip"]["reason"]
        == "user_disallowed_source_vs_quant_due_ram"
    )
    assert (
        "No-heavy source contract only"
        in result["diagnostics"]["api_cache_responses_contract"]["boundary"]
    )
    assert result["diagnostics"]["manual_sink_sdpa_clears_length_generation"] is False
    assert result["diagnostics"]["disable_sink_clears_length_generation"] is False
    assert "sink-kernel-only" in result["diagnostics"]["sink_boundary"]
    assert "mimo_jang2l_live_media_l2_missing" in result["blockers"]
    assert "mimo_jang2l_media_capability_downscoped_to_text" not in result["blockers"]
    assert "mimo_jang2l_media_capability_memory_gated" in result["blockers"]
    assert "mimo_jang2l_media_prefill_budget_blocked" not in result["blockers"]
    assert (
        "mimo_jang2l_post_media_working_set_pressure_blocked"
        not in result["blockers"]
    )


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
                        "content": "",
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
                        "content": "",
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
    assert evidence["block_disk_l2_restart_restore_pass"] is False
    assert evidence["block_disk_l2_restart_restore_reason"] == "l2_restart_probe_not_run"
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
    assert evidence["artifact_exactness_boundary"]["empty_visible_output"] is False
    assert "tool_required" in evidence["artifact_exactness_boundary"]["failed_labels"]


def test_mimo_all_local_smoke_evidence_classifies_empty_visible_output_separately():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "status": "probe_failed",
                "requests": [
                    {
                        "label": "mimo_structured_json_sentinel",
                        "code": 200,
                        "content": "",
                        "validation_failures": [
                            {
                                "label": "mimo_structured_json_sentinel",
                                "reason": "empty_visible",
                            }
                        ],
                    }
                ],
                "failures": [
                    {
                        "label": "mimo_structured_json_sentinel",
                        "reason": "empty_visible",
                    }
                ],
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["artifact_exactness_boundary"]["classification"] == (
        "empty_visible_output_after_generation_stop"
    )
    assert (
        evidence["artifact_exactness_boundary"][
            "parser_structure_valid_for_failed_rows"
        ]
        is False
    )
    assert evidence["artifact_exactness_boundary"]["empty_visible_output"] is True
    assert "empty visible output" in evidence["artifact_exactness_boundary"][
        "release_boundary"
    ]


def test_mimo_object_media_rows_clear_live_media_e2e_separately_from_exactness():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "status": "probe_failed",
                "requests": [
                    {
                        "label": "vl_blue_image",
                        "code": 200,
                        "content": "Blue",
                        "validation_failures": [],
                    },
                    {
                        "label": "vl_blue_image_repeat",
                        "code": 200,
                        "content": "Blue",
                        "validation_failures": [],
                    },
                    {
                        "label": "vl_red_image_changed",
                        "code": 200,
                        "content": "Red",
                        "validation_failures": [],
                    },
                    {
                        "label": "text_no_media_after_image",
                        "code": 200,
                        "content": "NONE",
                        "validation_failures": [],
                    },
                    {
                        "label": "vl_blue_video",
                        "code": 200,
                        "content": "Blue",
                        "validation_failures": [],
                    },
                    {
                        "label": "text_no_media_after_video",
                        "code": 200,
                        "content": "NONE",
                        "validation_failures": [],
                    },
                ],
                "failures": [
                    {
                        "label": "mimo_tool_required_sentinel",
                        "reason": "expected_tool_argument_missing",
                    }
                ],
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["live_media_e2e_pass"] is True
    assert evidence["live_media_e2e_status"] == "pass"
    assert evidence["live_media_e2e_labels"] == [
        "vl_blue_image",
        "vl_blue_image_repeat",
        "vl_red_image_changed",
        "text_no_media_after_image",
        "vl_blue_video",
        "text_no_media_after_video",
    ]
    assert evidence["artifact_exactness_pass"] is False


def test_mimo_all_local_smoke_evidence_accepts_disk_l2_restart_restore():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "pass",
        "results": [
            {
                "status": "pass",
                "requests": [],
                "failures": [],
                "cache_after": {
                    "body": {
                        "scheduler_stats": {
                            "batch_generator": {"generation_tps": 42.0}
                        }
                    }
                },
                "l2_restart": {
                    "summary": {
                        "status": "pass",
                        "pass": True,
                        "reason": "pass",
                        "disk_hits": 1,
                        "cache_hit_tokens": 64,
                    }
                },
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["block_disk_l2_restart_restore_pass"] is True
    assert evidence["block_disk_l2_restart_restore_status"] == "pass"
    assert evidence["block_disk_l2_restart_restore_reason"] == "pass"


def test_mimo_all_local_smoke_evidence_accepts_narrow_text_cache_repeat():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "probe_failed",
        "results": [
            {
                "status": "probe_failed",
                "failures": [
                    {
                        "label": "tool_required",
                        "reason": "expected_tool_argument_missing",
                    }
                ],
                "requests": [
                    {
                        "label": "text_cache_repeat_1",
                        "content": "ACK",
                        "cache_summary": {"cache_hit_tokens": 0},
                    },
                    {
                        "label": "text_cache_repeat_2",
                        "content": "ACK",
                        "usage": {
                            "prompt_tokens_details": {
                                "cached_tokens": 60,
                                "cache_detail": "paged",
                            }
                        },
                        "cache_summary": {
                            "cache_hit_tokens": 60,
                            "has_cache_hit": True,
                        },
                    },
                ],
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["text_cache_narrow_pass"] is True
    assert evidence["artifact_exactness_pass"] is False


def test_mimo_all_local_smoke_evidence_tracks_bundle_identity_for_media_l2():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "pass",
        "results": [
            {
                "name": "MiMo-V2.5-JANGTQ_2",
                "relative_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
                "status": "pass",
                "requests": [
                    {
                        "label": "vl_blue_image",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "vl_blue_image_repeat",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "vl_red_image_changed",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "text_no_media_after_image",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "vl_blue_video",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "text_no_media_after_video",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "audio_blue",
                        "code": 200,
                        "validation_failures": [],
                    },
                    {
                        "label": "text_no_media_after_audio",
                        "code": 200,
                        "validation_failures": [],
                    },
                ],
                "failures": [],
                "l2_restart": {
                    "summary": {
                        "status": "pass",
                        "pass": True,
                        "reason": "pass",
                    }
                },
                "cache_after": {
                    "body": {
                        "scheduler_stats": {
                            "batch_generator": {"generation_tps": 45.0}
                        }
                    }
                },
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["bundle_name"] == "MiMo-V2.5-JANGTQ_2"
    assert evidence["bundle_kind"] == "jangtq2"
    assert evidence["bundle_media_l2_coverage"] == {
        "jangtq2": {
            "bundle_name": "MiMo-V2.5-JANGTQ_2",
            "live_media_e2e_pass": True,
            "block_disk_l2_restart_cache_hit": True,
            "block_disk_l2_restart_restore_pass": True,
            "block_disk_l2_restart_visible_output_pass": True,
        },
        "jang2l": {
            "bundle_name": None,
            "live_media_e2e_pass": False,
            "block_disk_l2_restart_cache_hit": False,
            "block_disk_l2_restart_restore_pass": False,
            "block_disk_l2_restart_visible_output_pass": False,
        },
    }


def test_mimo_jangtq2_media_l2_transport_is_separate_from_visible_restore_pass():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    media_requests = [
        {"label": "vl_blue_image", "code": 200, "validation_failures": []},
        {"label": "vl_blue_image_repeat", "code": 200, "validation_failures": []},
        {"label": "vl_red_image_changed", "code": 200, "validation_failures": []},
        {"label": "text_no_media_after_image", "code": 200, "validation_failures": []},
        {"label": "vl_blue_video", "code": 200, "validation_failures": []},
        {"label": "text_no_media_after_video", "code": 200, "validation_failures": []},
        {"label": "audio_blue", "code": 200, "validation_failures": []},
        {"label": "text_no_media_after_audio", "code": 200, "validation_failures": []},
    ]
    smoke = {
        "status": "fail",
        "results": [
            {
                "name": "MiMo-V2.5-JANGTQ_2",
                "relative_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
                "status": "probe_failed",
                "requests": media_requests,
                "failures": [
                    {
                        "label": "text_cache_repeat_l2_restart",
                        "reason": "validation_failed",
                    }
                ],
                "l2_restart": {
                    "label": "text_cache_repeat_l2_restart",
                    "status": "completed",
                    "code": 200,
                    "content": "ACK plus prompt echo",
                    "usage": {
                        "prompt_tokens_details": {
                            "cached_tokens": 67,
                            "cache_detail": "paged+disk",
                        }
                    },
                    "cache_stats": {
                        "scheduler_stats": {
                            "cache_hit_tokens": 67,
                            "cache_hit_tokens_by_detail": {"paged+disk": 67},
                        },
                        "block_disk_cache": {"disk_hits": 4},
                    },
                    "summary": {
                        "status": "fail",
                        "pass": False,
                        "reason": "validation_failed",
                    },
                },
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["live_media_e2e_pass"] is True
    assert evidence["audio_waveform_e2e_pass"] is True
    assert evidence["block_disk_l2_restart_cache_hit"] is True
    assert evidence["block_disk_l2_restart_restore_pass"] is False
    assert evidence["bundle_media_l2_coverage"]["jangtq2"] == {
        "bundle_name": "MiMo-V2.5-JANGTQ_2",
        "live_media_e2e_pass": True,
        "block_disk_l2_restart_cache_hit": True,
        "block_disk_l2_restart_restore_pass": False,
        "block_disk_l2_restart_visible_output_pass": False,
    }


def test_mimo_all_local_smoke_evidence_reads_nested_row_bundle_identity():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "row": {
                    "name": "MiMo-V2.5-JANGTQ_2",
                    "relative_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
                },
                "status": "probe_failed",
                "requests": [
                    {"label": "vl_blue_image", "code": 200, "validation_failures": []},
                    {"label": "vl_blue_image_repeat", "code": 200, "validation_failures": []},
                    {"label": "vl_red_image_changed", "code": 200, "validation_failures": []},
                    {"label": "text_no_media_after_image", "code": 200, "validation_failures": []},
                    {"label": "vl_blue_video", "code": 200, "validation_failures": []},
                    {"label": "text_no_media_after_video", "code": 200, "validation_failures": []},
                    {"label": "audio_blue", "code": 200, "validation_failures": []},
                    {"label": "text_no_media_after_audio", "code": 200, "validation_failures": []},
                ],
                "failures": [],
                "l2_restart": {
                    "label": "text_cache_repeat_l2_restart",
                    "status": "completed",
                    "code": 200,
                    "content": "ACK",
                    "usage": {
                        "prompt_tokens_details": {
                            "cached_tokens": 67,
                            "cache_detail": "paged+disk",
                        }
                    },
                    "cache_stats": {
                        "scheduler_stats": {
                            "cache_hit_tokens": 67,
                            "cache_hit_tokens_by_detail": {"paged+disk": 67},
                        },
                        "block_disk_cache": {"disk_hits": 4},
                    },
                    "summary": {
                        "status": "pass",
                        "pass": True,
                        "reason": "pass",
                    },
                },
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["bundle_kind"] == "jangtq2"
    assert evidence["bundle_media_l2_coverage"]["jangtq2"] == {
        "bundle_name": "MiMo-V2.5-JANGTQ_2",
        "live_media_e2e_pass": True,
        "block_disk_l2_restart_cache_hit": True,
        "block_disk_l2_restart_restore_pass": True,
        "block_disk_l2_restart_visible_output_pass": True,
    }


def test_mimo_jang2l_l2_cache_hit_is_not_visible_restore_pass():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "name": "MiMo-V2.5-JANG_2L",
                "relative_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L",
                "status": "probe_failed",
                "requests": [],
                "failures": [
                    {
                        "label": "text_cache_repeat_l2_restart",
                        "reason": "validation_failed",
                    }
                ],
                "l2_restart": {
                    "label": "text_cache_repeat_l2_restart",
                    "status": "completed",
                    "code": 200,
                    "content": "",
                    "usage": {
                        "prompt_tokens_details": {
                            "cached_tokens": 54,
                            "cache_detail": "paged+disk",
                        }
                    },
                    "cache_stats": {
                        "scheduler_stats": {
                            "cache_hit_tokens": 54,
                            "cache_hit_tokens_by_detail": {"paged+disk": 54},
                        },
                        "block_disk_cache": {"disk_hits": 1},
                    },
                },
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["bundle_kind"] == "jang2l"
    assert evidence["block_disk_l2_restart_cache_hit"] is True
    assert evidence["block_disk_l2_restart_restore_pass"] is False
    assert evidence["bundle_media_l2_coverage"]["jang2l"] == {
        "bundle_name": "MiMo-V2.5-JANG_2L",
        "live_media_e2e_pass": False,
        "block_disk_l2_restart_cache_hit": True,
        "block_disk_l2_restart_restore_pass": False,
        "block_disk_l2_restart_visible_output_pass": False,
    }


def test_mimo_current_audit_points_jangtq2_at_latest_cache_cap_smoke():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    assert str(audit.ALL_LOCAL_SMOKE_ARTIFACT) == (
        "build/current-all-local-model-smoke-mimo-v25-jangtq2-current-source-textonly-l2-after-capability-fix-20260609/"
        "JANGQ_MiMo-V2.5-JANGTQ_2/result.json"
    )


def test_mimo_current_audit_points_switchglu_at_current_parity_proof():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    assert str(audit.SWITCHGLU_ARTIFACT) == (
        "build/current-mimo-v2-switchglu-selected-expert-parity-20260609.json"
    )


def test_mimo_current_audit_points_at_current_cache_and_classifier_artifacts():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    assert str(audit.DEFAULT_OUT) == (
        "build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json"
    )
    assert str(audit.NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT) == (
        "build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json"
    )


def test_mimo_current_audit_points_jang2l_at_latest_media_l2_release_smoke():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    assert str(audit.JANG2L_ALL_LOCAL_SMOKE_ARTIFACT) == (
        "build/current-all-local-model-smoke-mimo-v25-jang2l-live-refresh-20260608/"
        "summary.json"
    )


def test_mimo_audio_waveform_rows_clear_audio_gate_separately_from_image_video():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "status": "probe_failed",
                "requests": [
                    {
                        "label": "audio_blue",
                        "code": 200,
                        "content": "Blue.",
                        "validation_failures": [],
                    },
                    {
                        "label": "text_no_media_after_audio",
                        "code": 200,
                        "content": "NONE",
                        "validation_failures": [],
                    },
                ],
                "failures": [
                    {
                        "label": "mimo_tool_required_sentinel",
                        "reason": "expected_tool_argument_missing",
                    }
                ],
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["audio_waveform_e2e_pass"] is True
    assert evidence["audio_waveform_e2e_status"] == "pass"
    assert evidence["audio_waveform_e2e_classification"] == "audio_waveform_e2e_clear"
    assert evidence["audio_waveform_e2e_labels"] == [
        "audio_blue",
        "text_no_media_after_audio",
    ]
    assert evidence["live_media_e2e_pass"] is False
    assert evidence["artifact_exactness_pass"] is False


def test_mimo_audio_waveform_missing_payload_is_classified_as_bridge_gap():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "status": "probe_failed",
                "requests": [
                    {
                        "label": "audio_blue",
                        "code": 200,
                        "content": (
                            "Generation failed: unsupported media modality audio "
                            "for mimo_v2: raw audio reached the VLM processor, "
                            "but the processor returned no audio_codes."
                        ),
                        "validation_failures": [
                            {
                                "label": "audio_blue",
                                "reason": "audio_processor_payload_missing",
                            }
                        ],
                    },
                    {
                        "label": "text_no_media_after_audio",
                        "code": 200,
                        "content": "NONE",
                        "validation_failures": [],
                    },
                ],
                "failures": [
                    {
                        "label": "audio_blue",
                        "reason": "audio_processor_payload_missing",
                    }
                ],
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)

    assert evidence["audio_waveform_e2e_pass"] is False
    assert (
        evidence["audio_waveform_e2e_classification"]
        == "mimo_waveform_to_audio_codes_bridge_missing"
    )


def test_mimo_audit_accepts_fenced_json_without_semantic_repair():
    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    smoke = {
        "status": "fail",
        "results": [
            {
                "status": "probe_failed",
                "requests": [
                    {
                        "label": "mimo_structured_json_sentinel",
                        "code": 200,
                        "content": (
                            '```json\n'
                            '{"status":"ok","value":"B7-CAT-09","count":3}\n'
                            '```'
                        ),
                        "validation_failures": [
                            {
                                "label": "mimo_structured_json_sentinel",
                                "reason": "json_parse_failed",
                            }
                        ],
                    },
                    {
                        "label": "structured_json_exact",
                        "code": 200,
                        "content": (
                            '```json\n'
                            '{"status":"ok","value":"bluecat","count":3}\n'
                            '```'
                        ),
                        "validation_failures": [
                            {
                                "label": "structured_json_exact",
                                "reason": "json_exact_object_mismatch",
                                "expected": {
                                    "status": "ok",
                                    "value": "blue-cat",
                                    "count": 3,
                                },
                                "actual": {
                                    "status": "ok",
                                    "value": "bluecat",
                                    "count": 3,
                                },
                            }
                        ],
                    },
                ],
                "failures": [
                    {
                        "label": "mimo_structured_json_sentinel",
                        "reason": "json_parse_failed",
                    },
                    {
                        "label": "structured_json_exact",
                        "reason": "json_exact_object_mismatch",
                        "expected": {
                            "status": "ok",
                            "value": "blue-cat",
                            "count": 3,
                        },
                        "actual": {
                            "status": "ok",
                            "value": "bluecat",
                            "count": 3,
                        },
                    },
                ],
            }
        ],
    }

    evidence = audit._all_local_smoke_evidence(smoke)
    boundary = evidence["artifact_exactness_boundary"]

    assert boundary["classification"] == (
        "model_generated_literal_mutation_after_valid_parser_structure"
    )
    assert boundary["parser_structure_valid_for_failed_rows"] is True
    assert boundary["examples"][0]["actual"] == {
        "status": "ok",
        "value": "B7-CAT-09",
        "count": 3,
    }
    assert boundary["examples"][1]["actual"] == {
        "status": "ok",
        "value": "bluecat",
        "count": 3,
    }


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


def test_mimo_source_capability_detector_accepts_runtime_media(monkeypatch, tmp_path):
    import sys
    import types

    from tests.cross_matrix import run_mimo_v2_jang2l_current_audit as audit

    model_path = tmp_path / "MiMo-V2.5-JANGTQ_2"
    (model_path / "audio_tokenizer").mkdir(parents=True)
    (model_path / "audio_tokenizer" / "model.safetensors").write_bytes(b"stub")
    _write_json(
        model_path / "config.json",
        {
            "model_type": "mimo_v2",
            "vision_config": {},
            "audio_config": {},
            "processor_config": {"audio_token_id": 151669},
            "image_token_id": 151655,
            "video_token_id": 151656,
            "capabilities": {
                "modalities": ["text", "vision", "video", "audio"],
                "preserved_modalities": ["vision", "audio"],
                "unwired_modalities": [],
                "multimodal_status": "mimo_v2_multimodal_runtime",
            },
            "runtime": {"multimodal_mode": "mimo_v2_multimodal_runtime"},
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models.mimo_v2",
        types.SimpleNamespace(
            VisionModel=object,
            MiMoVisionPatchEmbed=object,
            MiMoVisionPatchMerger=object,
            MiMoVisionBlock=object,
            AudioModel=object,
            MiMoAudioTokenizer=object,
            load_mimo_audio_tokenizer_from_bundle=lambda *_args, **_kwargs: None,
        ),
    )

    evidence = audit._mimo_runtime_capabilities_from_source_detector(model_path)

    assert evidence["exists"] is True
    assert evidence["runtime_capabilities_safe"] is True
    assert evidence["runtime_media_supported"] is True
    assert evidence["status_by_modality"]["image"] == "runtime_supported"
    assert evidence["status_by_modality"]["audio"] == "runtime_supported"

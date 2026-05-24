import json
from pathlib import Path


def _write_json(root: Path, rel: str, obj: dict) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_passing_base_artifacts(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "build/current-dsv4-cache-proof-digest-20260521.json",
        {
            "checks": {
                "persistedDefaultOn": True,
                "launchHasDsv4EnablePrefix": True,
                "launchHasUsePagedCache": True,
                "launchHasBlockDisk": True,
                "launchNoDisablePrefix": True,
                "hotFasterTtft": True,
                "sameProcessHitDsv4": True,
                "blockDiskWrite": True,
                "restartL2DiskHit": True,
                "restartDsv4CacheHit": True,
            },
            "timings": {"cold_ttft_sec": 15.0, "hot_ttft_sec": 1.0},
            "stats_after_hot": {"block_disk_cache": {"disk_writes": 3}},
            "stats_after_restart": {"block_disk_cache": {"disk_hits": 3}},
        },
    )
    _write_json(
        tmp_path,
        "build/dev-ui-dsv4-live-cache-proof-20260521/result.json",
        {
            "before": {
                "body": {
                    "native_cache": {
                        "cache_type": "native_composite",
                        "components": ["swa_local", "csa_compressed_pool", "hca_compressed_pool"],
                        "prefix": True,
                        "paged": True,
                        "block_disk_l2": True,
                        "generic_turboquant_kv": {"enabled": False},
                    }
                }
            },
            "turn1": {"elapsed_sec": 25.0},
            "turn2": {
                "elapsed_sec": 1.0,
                "body": {"usage": {"prompt_tokens_details": {"cached_tokens": 998}}},
            },
        },
    )
    _write_json(
        tmp_path,
        "build/dev-ui-dsv4-live-tool-proof-20260521/result.json",
        {
            "round1": {
                "body": {
                    "output": [
                        {"type": "function_call", "name": "list_directory", "arguments": "{\"path\":\".\"}"}
                    ]
                }
            },
            "round2": {"body": {"output": [], "output_text": "DONE"}},
        },
    )
    _write_json(
        tmp_path,
        "build/v1546-current-bundled-dsv4-two-tool-proof-20260521001426/result.json",
        {
            "rounds": [
                {"executed_tools": [{"name": "list_directory"}]},
                {"executed_tools": [{"name": "write_file"}]},
                {"executed_tools": [], "output_text": "DONE"},
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-default-cache-tool-loop/result.json",
        {
            "cmd": [
                "python",
                "-m",
                "vmlx_engine.cli",
                "serve",
                "/models/dsv4",
                "--dsv4-enable-prefix-cache",
                "--use-paged-cache",
                "--enable-block-disk-cache",
                "--tool-call-parser",
                "dsml",
            ],
            "health": {
                "native_cache": {
                    "cache_type": "native_composite",
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                    "generic_turboquant_kv": {"enabled": False},
                }
            },
            "rounds": [
                {
                    "executed_tools": [{"name": "list_directory"}],
                    "usage": {
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_detail": "",
                        }
                    },
                },
                {
                    "executed_tools": [{"name": "write_file"}],
                    "usage": {
                        "input_tokens_details": {
                            "cached_tokens": 128,
                            "cache_detail": "paged+dsv4",
                        }
                    },
                },
                {
                    "executed_tools": [],
                    "output_text": "DONE",
                    "usage": {
                        "input_tokens_details": {
                            "cached_tokens": 192,
                            "cache_detail": "paged+dsv4",
                        }
                    },
                },
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/v1546-dsv4-app-tool-cap-nocache-proof-20260521090706/summary.json",
        {
            "allPassed": True,
            "checks": {
                "maxToolIterationsPersisted": True,
                "sawLimitMessage": True,
                "executedExactlyOneRealTool": True,
                "capFileNotWritten": True,
            },
        },
    )
    _write_json(
        tmp_path,
        "build/dev-ui-smoke-20260521/summary.json",
        {
            "visible_assertions": {
                "server_default_max_output_visible": True,
                "max_context_visible": True,
                "generation_defaults_visible": True,
            },
            "cli_preview_assertions": {
                "has_max_tokens_flag_after_reset": False,
                "has_max_prompt_tokens_flag_after_reset": False,
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-static-cache-architecture-audit-full-qwen-hybrid-20260521.json",
        {
            "rows": [
                {
                    "static": {
                        "id": row_id,
                        "exists": True,
                        "cache_profile_expected": expected,
                        "registry": {"cache_type": registry_cache},
                        "issues": [],
                    }
                }
                for row_id, expected, registry_cache in (
                    ("dsv4_jang_local", "dsv4_composite", "kv"),
                    ("minimax_m27_tq_k", "default", "kv"),
                    ("qwen36_dense_mxfp4", "hybrid_ssm", "hybrid"),
                    ("qwen36_dense_jang", "hybrid_ssm", "hybrid"),
                    ("zaya_vl_mxfp4", "zaya_cca", "hybrid"),
                    ("nemotron_omni_tq2", "hybrid_ssm", "hybrid"),
                    ("ling_flash_tq", "hybrid_ssm", "hybrid"),
                )
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-long-context-proof-digest-20260521.json",
        {"status": "review", "notes": ["identifier corruption still open"]},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-matrix-20260523.json",
        {
            "status": "pass",
            "probes": [
                {
                    "id": "copy_identifiers_only",
                    "status": "pass",
                    "required_identifier_counts": {
                        "THREE.Scene": 1,
                        "THREE.WebGLRenderer": 1,
                        "THREE.PerspectiveCamera": 1,
                        "THREE.BoxGeometry": 1,
                        "THREE.MeshBasicMaterial": 1,
                    },
                    "has_markdown_fence": False,
                    "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTHREE.BoxGeometry\nTHREE.MeshBasicMaterial",
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-installed-tokenizer-roundtrip-20260523.json",
        {
            "load_method": "AutoTokenizer.from_pretrained",
            "tokenizer_class": "TokenizersBackend",
            "all_roundtrip_exact": True,
            "rows": [
                {
                    "input": "THREE.PerspectiveCamera",
                    "ids": [8840, 21679, 5497, 387, 126614, 65184],
                    "tokens": ["TH", "REE", ".P", "ers", "pective", "Camera"],
                    "decoded": "THREE.PerspectiveCamera",
                    "roundtrip_exact": True,
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-logprobs-copy-20260523.json",
        {
            "status": "pass",
            "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera",
            "identifier_counts": {"THREE.PerspectiveCamera": 1},
            "logprob_entry_count": 6,
            "response": {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "TH", "logprob": -0.01, "top_logprobs": []},
                                {"token": "REE", "logprob": -0.01, "top_logprobs": []},
                                {"token": ".P", "logprob": -0.01, "top_logprobs": []},
                                {"token": "ers", "logprob": -0.01, "top_logprobs": []},
                                {"token": "pective", "logprob": -0.01, "top_logprobs": []},
                                {"token": "Camera", "logprob": -0.01, "top_logprobs": []},
                            ]
                        }
                    }
                ]
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-logprob-context-matrix-20260523.json",
        {
            "status": "pass",
            "target": "THREE.PerspectiveCamera",
            "probes": [
                {
                    "id": "isolated_identifier",
                    "content": "THREE.PerspectiveCamera",
                    "target_count": 1,
                    "has_wrong_perscpective": False,
                    "tokens": ["TH", "REE", ".P", "ers", "pective", "Camera"],
                    "wrong_c_after_pers_events": [],
                },
                {
                    "id": "list_copy",
                    "content": "THREE.PerspectiveCamera",
                    "target_count": 1,
                    "has_wrong_perscpective": False,
                    "tokens": ["TH", "REE", ".P", "ers", "pective", "Camera"],
                    "wrong_c_after_pers_events": [],
                },
                {
                    "id": "constructor_sentence",
                    "content": "new THREE.PerspectiveCamera(45, 1, 0.1, 1000)",
                    "target_count": 1,
                    "has_wrong_perscpective": False,
                    "tokens": ["new", " THREE", ".P", "ers", "pective", "Camera"],
                    "wrong_c_after_pers_events": [],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-cache-context-identifier-probe-20260523.json",
        {
            "status": "pass",
            "health_before": {
                "native_cache": {
                    "pool_quant": {"enabled": True, "env": "1"},
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                    "generic_turboquant_kv": {"enabled": False},
                },
                "kv_cache_quantization": {"enabled": False},
            },
            "results": [
                {
                    "name": "list_plain",
                    "status": "pass",
                    "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTHREE.BoxGeometry\nTHREE.MeshBasicMaterial",
                    "expected_identifiers_missing": [],
                    "corrupt_identifier_tokens": [],
                },
                {
                    "name": "list_unique_prefix",
                    "status": "pass",
                    "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTHREE.BoxGeometry\nTHREE.MeshBasicMaterial",
                    "expected_identifiers_missing": [],
                    "corrupt_identifier_tokens": [],
                },
                {
                    "name": "constructor_unique_prefix",
                    "status": "pass",
                    "content": "new THREE.PerspectiveCamera(45, 1, 0.1, 1000)",
                    "expected_identifiers_missing": [],
                    "corrupt_identifier_tokens": [],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "tests/test_engine_audit.py",
        {"fixture": "engine-audit"},
    )
    _write_json(
        tmp_path,
        "vmlx_engine/server.py",
        {"fixture": "server"},
    )
    _write_json(
        tmp_path,
        "vmlx_engine/tool_parsers/dsml_tool_parser.py",
        {"fixture": "dsml-parser"},
    )
    _write_json(
        tmp_path,
        "tests/test_batching.py",
        {"fixture": "batching"},
    )
    _write_json(
        tmp_path,
        "tests/test_mllm_scheduler_cache.py",
        {"fixture": "mllm-cache"},
    )
    _write_json(
        tmp_path,
        "tests/test_tq_disk_cache.py",
        {"fixture": "tq-disk"},
    )
    _write_json(
        tmp_path,
        "tests/test_dsml_tool_parser.py",
        {"fixture": "dsml-tests"},
    )
    _write_json(
        tmp_path,
        "tests/test_tool_format.py",
        {"fixture": "tool-format"},
    )
    source_hashes = {
        rel: _sha256(tmp_path / rel)
        for rel in (
            "vmlx_engine/server.py",
            "vmlx_engine/tool_parsers/dsml_tool_parser.py",
            "tests/test_engine_audit.py",
            "tests/test_batching.py",
            "tests/test_mllm_scheduler_cache.py",
            "tests/test_tq_disk_cache.py",
            "tests/test_dsml_tool_parser.py",
            "tests/test_tool_format.py",
        )
    }
    _write_json(
        tmp_path,
        "build/current-api-cache-contract-proof-20260521.json",
        {
            "status": "pass",
            "checks": {
                "openai_chat_sampling_kwargs": True,
                "responses_sampling_kwargs": True,
                "anthropic_bundle_defaults": True,
                "ollama_adapter_surface": True,
                "dsv4_native_cache_status": True,
                "dsv4_dsml_parser_residue_rejection": True,
                "dsv4_dsml_valid_tool_call_preserved": True,
                "dsv4_suppressed_tool_markup_not_stored": True,
                "zaya_typed_cca_status": True,
                "hybrid_ssm_partial_reuse": True,
                "turboquant_kv_runtime_contract": True,
                "turboquant_disk_roundtrip": True,
                "no_generic_tq_on_hybrid_ssm": True,
            },
            "source_hashes": source_hashes,
        },
    )
    for rel in (
        "panel/src/main/sessions.ts",
        "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
        "panel/src/renderer/src/components/chat/ChatSettings.tsx",
        "panel/src/shared/dsv4Env.ts",
        "panel/src/shared/cacheControlPolicy.ts",
        "panel/tests/settings-flow.test.ts",
        "panel/tests/dsv4-env.test.ts",
        "panel/tests/cache-control-policy.test.ts",
    ):
        _write_json(tmp_path, rel, {"fixture": rel})
    panel_source_hashes = {
        rel: _sha256(tmp_path / rel)
        for rel in (
            "panel/src/main/sessions.ts",
            "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
            "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
            "panel/src/renderer/src/components/chat/ChatSettings.tsx",
            "panel/src/shared/dsv4Env.ts",
            "panel/src/shared/cacheControlPolicy.ts",
            "panel/tests/settings-flow.test.ts",
            "panel/tests/dsv4-env.test.ts",
            "panel/tests/cache-control-policy.test.ts",
        )
    }
    _write_json(
        tmp_path,
        "build/current-panel-settings-contract-proof-20260521.json",
        {
            "status": "pass",
            "checks": {
                "dsv4_default_native_prefix_on": True,
                "dsv4_explicit_prefix_off_disables_native_flags": True,
                "dsv4_l2_explicit_off_preserves_prefix": True,
                "dsv4_generic_kv_flags_suppressed": True,
                "max_output_context_cli_split": True,
                "chat_max_output_is_per_chat_override": True,
                "non_dsv4_cache_toggles_preserved": True,
                "i18n_max_output_context_copy": True,
                "panel_typecheck": True,
            },
            "source_hashes": panel_source_hashes,
        },
    )
    from tests.cross_matrix.summarize_objective_proof import (
        MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS,
        MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES,
    )

    for rel in MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES:
        if not (tmp_path / rel).exists():
            _write_json(tmp_path, rel, {"fixture": rel})
    _write_json(
        tmp_path,
        "build/current-max-output-context-contract-20260521.json",
        {
            "status": "pass",
            "checks": {
                key: True
                for key in MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS
            },
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES
            },
        },
    )
    from tests.cross_matrix.summarize_objective_proof import (
        MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS,
        MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
        MODEL_FAMILY_CONTRACT_CHECKS,
        MODEL_FAMILY_SOURCE_HASH_FILES,
        PARSER_REGISTRY_CONTRACT_CHECKS,
        PARSER_REGISTRY_SOURCE_HASH_FILES,
    )

    for rel in (
        *MODEL_FAMILY_SOURCE_HASH_FILES,
        *PARSER_REGISTRY_SOURCE_HASH_FILES,
        *MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
    ):
        if not (tmp_path / rel).exists():
            _write_json(tmp_path, rel, {"fixture": rel})
    _write_json(
        tmp_path,
        "build/current-model-family-detection-contract-20260521.json",
        {
            "status": "pass",
            "checks": {key: True for key in MODEL_FAMILY_CONTRACT_CHECKS},
            "failed": [],
            "missing_rows": [],
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in MODEL_FAMILY_SOURCE_HASH_FILES
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-parser-registry-contract-20260521.json",
        {
            "status": "pass",
            "checks": {key: True for key in PARSER_REGISTRY_CONTRACT_CHECKS},
            "failed": [],
            "missing_markers": [],
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in PARSER_REGISTRY_SOURCE_HASH_FILES
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-model-artifact-format-contract-20260521.json",
        {
            "status": "pass",
            "checks": {
                key: True
                for key in MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS
            },
            "failed": [],
            "missing_markers": [],
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES
            },
        },
    )
    from tests.cross_matrix.summarize_objective_proof import (
        GENERATION_DEFAULTS_CONTRACT_CHECKS,
        GENERATION_DEFAULTS_SOURCE_HASH_FILES,
        NATIVE_MTP_CONTRACT_CHECKS,
        NATIVE_MTP_SOURCE_HASH_FILES,
        VL_MEDIA_CONTRACT_CHECKS,
        VL_MEDIA_SOURCE_HASH_FILES,
    )

    for rel in (
        *GENERATION_DEFAULTS_SOURCE_HASH_FILES,
        *NATIVE_MTP_SOURCE_HASH_FILES,
        *VL_MEDIA_SOURCE_HASH_FILES,
    ):
        if not (tmp_path / rel).exists():
            _write_json(tmp_path, rel, {"fixture": rel})
    _write_json(
        tmp_path,
        "build/current-generation-defaults-contract-20260521.json",
        {
            "status": "pass",
            "checks": {
                key: True
                for key in GENERATION_DEFAULTS_CONTRACT_CHECKS
            },
            "failed": [],
            "missing_markers": [],
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in GENERATION_DEFAULTS_SOURCE_HASH_FILES
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-native-mtp-contract-20260521.json",
        {
            "status": "pass",
            "checks": {key: True for key in NATIVE_MTP_CONTRACT_CHECKS},
            "failed": [],
            "missing_markers": [],
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in NATIVE_MTP_SOURCE_HASH_FILES
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-vl-media-cache-contract-20260521.json",
        {
            "status": "pass",
            "checks": {key: True for key in VL_MEDIA_CONTRACT_CHECKS},
            "failed": [],
            "missing_markers": [],
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in VL_MEDIA_SOURCE_HASH_FILES
            },
        },
    )
    speed_base = {
        "created_at": 1779493036.61303,
        "results": [
            {
                "name": "qwen27_jang4m",
                "status": "pass",
                "notes": [],
                "runtime_wheels": {
                    "mlx": ["cp312-cp312-macosx_26_0_arm64"],
                    "mlx-metal": ["py3-none-macosx_26_0_arm64"],
                },
                "pp_rows": [
                    {"target_tokens": 1024, "pp_wall_tok_s": 910.1, "loopish": False},
                    {"target_tokens": 4096, "pp_wall_tok_s": 914.5, "loopish": False},
                    {"target_tokens": 16384, "pp_wall_tok_s": 779.4, "loopish": False},
                ],
            }
        ],
    }
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-source-keepalloc-20260522.json",
        speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-packaged-tahoe-dmg-20260522.json",
        speed_base,
    )
    mtp_speed_base = {
        "created_at": 1779498428.0,
        "results": [
            {
                "name": "qwen27_jang4m_mtp",
                "status": "pass",
                "notes": [],
                "registry": {
                    "family_name": "qwen3_5",
                    "cache_type": "hybrid",
                    "is_mllm": True,
                    "tool_parser": "qwen",
                    "reasoning_parser": "qwen3",
                },
                "pp_rows": [
                    {"target_tokens": 1024, "pp_wall_tok_s": 710.1, "loopish": False},
                    {"target_tokens": 4096, "pp_wall_tok_s": 714.5, "loopish": False},
                    {"target_tokens": 16384, "pp_wall_tok_s": 679.4, "loopish": False},
                ],
                "greedy_topk0": {
                    "decode_tps_wall": 33.4,
                    "loopish": False,
                    "completion_tokens": 320,
                },
                "health_after": {
                    "scheduler": {
                        "batch_generator": {
                            "last_native_mtp": {
                                "final_depth": 1,
                                "accepted_tokens": 169,
                                "drafted_tokens": 180,
                                "acceptance_rate": 0.9388888889,
                            }
                        }
                    }
                },
            }
        ],
    }
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json",
        speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json",
        {
            "results": [
                {
                    "health_after": {
                        "scheduler": {
                            "batch_generator": {
                                "last_prefill_trace": {
                                    "forward_ms": 10.7,
                                    "logits_eval_ms": 204.2,
                                }
                            }
                        }
                    }
                }
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-no-prefix-logits-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-hybrid-long-prefix-split-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-kvnone-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-route-trace-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-default-after-norm-shift-20260523.json",
        mtp_speed_base,
    )
    _write_json(
        tmp_path,
        "build/current-qwen-forward-path-ab-1024-vlm-loader-20260523.json",
        {
            "comparison": [
                {
                    "prompt_tokens": 1024,
                    "text_pp_tok_s": 940.67,
                    "vlm_pp_tok_s": 301.68,
                    "text_over_vlm": 3.11,
                }
            ],
            "generation_defaults_policy": "raw-forward-only; no sampler or generation kwargs are accepted",
        },
    )
    _write_json(
        tmp_path,
        "build/current-qwen-forward-path-ab-4096-vlm-loader-20260523.json",
        {
            "comparison": [
                {
                    "prompt_tokens": 4096,
                    "text_pp_tok_s": 930.43,
                    "vlm_pp_tok_s": 296.75,
                    "text_over_vlm": 3.13,
                }
            ],
            "generation_defaults_policy": "raw-forward-only; no sampler or generation kwargs are accepted",
        },
    )
    _write_json(
        tmp_path,
        "build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-20260523/result.json",
        {
            "speedup_vs_baseline": 1.83,
            "output_equivalence": {
                "all_content_equal": True,
                "all_full_text_equal": True,
            },
            "rows": [
                {
                    "label": "baseline_no_mtp",
                    "summary": {"mean_wall_tok_s": 25.61},
                },
                {
                    "label": "native_mtp",
                    "summary": {"mean_wall_tok_s": 46.96},
                    "mtp_stats": {
                        "totals": {
                            "accepted_tokens": 218,
                            "drafted_tokens": 228,
                            "acceptance_rate": 0.956,
                        }
                    },
                },
            ],
        },
    )


def test_objective_proof_digest_keeps_dsv4_long_quality_open(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    assert rows["DSV4 long-output/code/file-generation quality is release-cleared"][
        "status"
    ] == "open"
    assert rows["DSV4 Flash prefix/paged/L2 cache is enabled by default from app launch"][
        "status"
    ] == "pass"
    assert sum(item["status"] == "open" for item in digest["requirements"]) == 1


def test_objective_proof_digest_surfaces_current_dsv4_identifier_canary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-canary-20260523.json",
        {
            "status": "review",
            "elapsed_sec": 62.352,
            "required_identifier_counts": {
                "THREE.Scene": 0,
                "THREE.WebGLRenderer": 1,
            },
            "bad_patterns": [],
            "has_markdown_fence": True,
            "content": "```javascript\nconst scene = new THREE.ScScene();\n```",
            "response": {
                "usage": {
                    "prompt_tokens": 81,
                    "completion_tokens": 52,
                    "total_tokens": 133,
                },
                "choices": [{"finish_reason": "stop"}],
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    canary = quality["details"]["current_installed_identifier_canary"]
    assert quality["status"] == "open"
    assert canary["status"] == "review"
    assert canary["required_identifier_counts"]["THREE.Scene"] == 0
    assert canary["has_markdown_fence"] is True
    assert canary["finish_reason"] == "stop"
    assert canary["usage"]["completion_tokens"] == 52
    assert "THREE.ScScene" in canary["content"]


def test_objective_proof_digest_prefers_strict_dsv4_identifier_canary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-canary-20260523.json",
        {
            "status": "review",
            "content": "old stale canary",
            "response": {"usage": {"completion_tokens": 1}, "choices": [{"finish_reason": "stop"}]},
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json",
        {
            "status": "fail",
            "model": "/models/DeepSeek-V4-Flash-JANGTQ-K",
            "env_overrides": {"DSV4_POOL_QUANT": "0"},
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "http_code": 200,
                    "finish": "stop",
                    "exact": False,
                    "corrupt_patterns": ["PPerspectiveCamera"],
                    "missing": ["THREE.PerspectiveCamera"],
                    "decode_tps_wall": 15.304,
                    "prompt_tokens": 47,
                    "completion_tokens": 32,
                    "content": "THREE.Scene\nTHREE.PPerspectiveCamera",
                }
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    canary = quality["details"]["current_installed_identifier_canary"]
    assert quality["status"] == "open"
    assert canary["artifact"].endswith("route-mode-code-exactness-20260524.json")
    assert canary["status"] == "fail"
    assert canary["env"]["DSV4_POOL_QUANT"] == "0"
    assert canary["failed_cases"] == ["chat_off"]
    assert canary["case_summaries"][0]["decode_tok_s_wall"] == 15.304
    assert canary["case_summaries"][0]["missing_identifiers"] == [
        "THREE.PerspectiveCamera"
    ]


def test_objective_proof_digest_surfaces_dsv4_prompt_rail_exactness_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-thinking-on-responses-controls.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer", "THREE.MeshBasicMaterial"],
                    "corrupt_patterns": ["WebWebGLRenderer", "Three.MeshBasicMaterial"],
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "prompt_endswith_assistant_think_open": False,
                        "prompt_endswith_assistant_think_close": True,
                        "prompt_tail": "renderer.render(scene, camera);<｜Assistant｜></think>",
                    },
                },
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "prompt_endswith_assistant_think_open": False,
                        "prompt_endswith_assistant_think_close": True,
                    },
                },
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": False,
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": True,
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "prompt_endswith_assistant_think_open": True,
                        "prompt_endswith_assistant_think_close": False,
                    },
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "exact": False,
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": True,
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "prompt_endswith_assistant_think_open": True,
                        "prompt_endswith_assistant_think_close": False,
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer", "THREE.MeshBasicMaterial"],
                    "corrupt_patterns": ["WebWebGLRenderer", "Three.MeshBasicMaterial"],
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "prompt_endswith_assistant_think_open": False,
                        "prompt_endswith_assistant_think_close": True,
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "prompt_endswith_assistant_think_open": False,
                        "prompt_endswith_assistant_think_close": True,
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": False,
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": True,
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "prompt_endswith_assistant_think_open": True,
                        "prompt_endswith_assistant_think_close": False,
                    },
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "exact": False,
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": True,
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "prompt_endswith_assistant_think_open": True,
                        "prompt_endswith_assistant_think_close": False,
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260524-thinking-on-responses-controls.json",
        {
            "status": "dry_run",
            "case_count": 9,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "request_overrides": {"enable_thinking": False, "max_tokens": 512},
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                },
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "request_overrides": {
                        "enable_thinking": False,
                        "max_tokens": 512,
                        "repetition_penalty": 1.0,
                    },
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "request_overrides": {"enable_thinking": False, "max_output_tokens": 512},
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "request_overrides": {
                        "enable_thinking": False,
                        "max_output_tokens": 512,
                        "repetition_penalty": 1.0,
                    },
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                },
                {
                    "name": "chat_on",
                    "route": "chat",
                    "request_overrides": {"enable_thinking": True, "max_tokens": 512},
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_open"},
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "request_overrides": {
                        "enable_thinking": True,
                        "max_tokens": 512,
                        "repetition_penalty": 1.0,
                    },
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_open"},
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "request_overrides": {"enable_thinking": True, "max_output_tokens": 512},
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_open"},
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "request_overrides": {
                        "enable_thinking": True,
                        "max_output_tokens": 512,
                        "repetition_penalty": 1.0,
                    },
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_open"},
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "request_overrides": {"max_tokens": 220},
                    "prompt_diagnostics": {"assistant_suffix_kind": "none"},
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_installed_prompt_rail_exactness_probe"]
    assert quality["status"] == "open"
    assert probe["status"] == "fail"
    assert probe["artifact"].endswith("thinking-on-responses-controls.json")
    assert probe["failed_thinking_closed_cases"] == [
        "chat_off",
        "chat_off_rep1",
        "responses_off",
        "responses_off_rep1",
    ]
    assert probe["rep1_still_corrupt_patterns"] == {
        "chat_off_rep1": ["WebWebGLRenderer"],
        "responses_off_rep1": ["WebWebGLRenderer"],
    }
    assert probe["thinking_open_cases_without_identifier_corruption"] == [
        "chat_on",
        "chat_on_rep1",
        "responses_on",
        "responses_on_rep1",
    ]
    assert probe["dry_run_status"] == "dry_run"
    assert probe["dry_run_case_count"] == 9
    assert probe["dry_run_suffixes"]["legacy_completion_raw"] == "none"
    assert probe["dry_run_suffixes"]["responses_on"] == "thinking_open"
    assert probe["case_summaries"][0]["assistant_suffix_kind"] == "thinking_closed"
    assert probe["case_summaries"][0]["prompt_tail"].endswith("</think>")


def test_objective_proof_digest_surfaces_current_dsv4_identifier_matrix(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-matrix-20260523.json",
        {
            "status": "review",
            "probes": [
                {
                    "id": "copy_identifiers_only",
                    "status": "review",
                    "elapsed_sec": 10.119,
                    "required_identifier_counts": {
                        "THREE.Scene": 1,
                        "THREE.WebGLRenderer": 1,
                        "THREE.PerspectiveCamera": 0,
                    },
                    "has_markdown_fence": False,
                    "content": "THREE.PerscpectiveCamera",
                },
                {
                    "id": "single_line_threejs_code",
                    "status": "review",
                    "elapsed_sec": 23.797,
                    "required_identifier_counts": {
                        "THREE.Scene": 1,
                        "THREE.WebGLRenderer": 0,
                        "THREE.PerspectiveCamera": 0,
                    },
                    "has_markdown_fence": True,
                    "content": "new THREE.WebWebGLRenderer(); new THREE.PersPerspectiveCamera();",
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    matrix = quality["details"]["current_installed_identifier_matrix"]
    assert quality["status"] == "open"
    assert matrix["status"] == "review"
    assert matrix["probe_count"] == 2
    assert matrix["failed_probe_ids"] == [
        "copy_identifiers_only",
        "single_line_threejs_code",
    ]
    assert matrix["copy_task_failed"] is True
    assert matrix["code_task_failed"] is True
    assert "THREE.PerscpectiveCamera" in matrix["probe_summaries"][0]["content"]
    assert "THREE.WebWebGLRenderer" in matrix["probe_summaries"][1]["content"]


def test_objective_proof_digest_surfaces_dsv4_tokenizer_roundtrip_boundary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-installed-tokenizer-roundtrip-20260523.json",
        {
            "load_method": "AutoTokenizer.from_pretrained",
            "tokenizer_class": "TokenizersBackend",
            "all_roundtrip_exact": True,
            "rows": [
                {
                    "input": "THREE.PerspectiveCamera",
                    "ids": [8840, 21679, 5497, 387, 126614, 65184],
                    "tokens": ["TH", "REE", ".P", "ers", "pective", "Camera"],
                    "decoded": "THREE.PerspectiveCamera",
                    "roundtrip_exact": True,
                },
                {
                    "input": "THREE.PerscpectiveCamera",
                    "ids": [8840, 21679, 5497, 387, 69, 126614, 65184],
                    "tokens": ["TH", "REE", ".P", "ers", "c", "pective", "Camera"],
                    "decoded": "THREE.PerscpectiveCamera",
                    "roundtrip_exact": True,
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    tokenizer = quality["details"]["current_installed_tokenizer_roundtrip"]
    assert quality["status"] == "open"
    assert tokenizer["status"] == "pass"
    assert tokenizer["all_roundtrip_exact"] is True
    assert tokenizer["tokenizer_class"] == "TokenizersBackend"
    assert tokenizer["row_count"] == 2
    assert tokenizer["failed_inputs"] == []
    assert tokenizer["rows"][0]["tokens"] == ["TH", "REE", ".P", "ers", "pective", "Camera"]
    assert tokenizer["rows"][1]["tokens"] == ["TH", "REE", ".P", "ers", "c", "pective", "Camera"]


def test_objective_proof_digest_surfaces_dsv4_logprob_corruption_boundary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-logprobs-copy-20260523.json",
        {
            "status": "pass",
            "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerscpectiveCamera",
            "identifier_counts": {"THREE.PerspectiveCamera": 0},
            "logprob_entry_count": 18,
            "response": {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "TH", "logprob": -0.01, "top_logprobs": []},
                                {"token": "REE", "logprob": -0.01, "top_logprobs": []},
                                {"token": ".P", "logprob": -0.01, "top_logprobs": []},
                                {"token": "ers", "logprob": -0.002, "top_logprobs": []},
                                {
                                    "token": "c",
                                    "logprob": -0.549,
                                    "top_logprobs": [
                                        {"token": "c", "logprob": -0.549},
                                        {"token": "pective", "logprob": -2.065},
                                    ],
                                },
                                {"token": "pective", "logprob": -0.048, "top_logprobs": []},
                                {"token": "Camera", "logprob": -0.002, "top_logprobs": []},
                            ]
                        }
                    }
                ]
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    logprobs = quality["details"]["current_installed_logprob_copy_probe"]
    assert quality["status"] == "open"
    assert logprobs["status"] == "review"
    assert logprobs["wrong_token"] == "c"
    assert logprobs["correct_token"] == "pective"
    assert logprobs["wrong_token_rank"] == 1
    assert logprobs["correct_token_rank"] == 2
    assert logprobs["model_preferred_wrong_token"] is True
    assert logprobs["tokens"] == ["TH", "REE", ".P", "ers", "c", "pective", "Camera"]


def test_objective_proof_digest_surfaces_dsv4_logprob_context_matrix(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-logprob-context-matrix-20260523.json",
        {
            "status": "review",
            "target": "THREE.PerspectiveCamera",
            "probes": [
                {
                    "id": "isolated_identifier",
                    "content": "THREE.PerspectiveCamera",
                    "target_count": 1,
                    "has_wrong_perscpective": False,
                    "tokens": ["TH", "REE", ".P", "ers", "pective", "Camera"],
                    "wrong_c_after_pers_events": [],
                },
                {
                    "id": "list_copy",
                    "content": "THREE.PerscpectiveCamera",
                    "target_count": 0,
                    "has_wrong_perscpective": True,
                    "tokens": ["TH", "REE", ".P", "ers", "c", "pective", "Camera"],
                    "wrong_c_after_pers_events": [
                        {
                            "token": "c",
                            "wrong_token_rank": 1,
                            "correct_token_rank": 2,
                            "model_preferred_wrong_token": True,
                        }
                    ],
                },
                {
                    "id": "constructor_sentence",
                    "content": "new THREE.PermpectiveCamera(45, 1, 0.1, 1000)",
                    "target_count": 0,
                    "has_wrong_perscpective": False,
                    "tokens": ["new", " THREE", ".P", "erm", "pective", "Camera"],
                    "wrong_c_after_pers_events": [],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    matrix = quality["details"]["current_installed_logprob_context_matrix"]
    assert quality["status"] == "open"
    assert matrix["status"] == "review"
    assert matrix["isolated_identifier_passed"] is True
    assert matrix["list_copy_failed"] is True
    assert matrix["constructor_sentence_failed"] is True
    assert matrix["context_sensitive_identifier_failure"] is True
    assert matrix["probe_summaries"]["constructor_sentence"]["content"].startswith("new THREE.PermpectiveCamera")


def test_objective_proof_digest_surfaces_dsv4_unique_prefix_identifier_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-cache-context-identifier-probe-20260523.json",
        {
            "status": "review",
            "health_before": {
                "native_cache": {
                    "pool_quant": {"enabled": True, "env": "1"},
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {"enabled": False},
            },
            "health_after": {
                "cache": {
                    "scheduler_cache": {
                        "cache_hits": 2,
                        "disk_hits": 1,
                    }
                }
            },
            "results": [
                {
                    "name": "list_plain",
                    "status": "review",
                    "content": "THREE.Scene\nTHIVE.WebGLRenderer\nTHREE.PerspectiveCamera",
                    "expected_identifiers_missing": ["THREE.WebGLRenderer"],
                    "corrupt_identifier_tokens": ["THIVE.WebGLRenderer"],
                },
                {
                    "name": "list_unique_prefix",
                    "status": "review",
                    "content": "THREE.Scene\nTHUEE.WebGLRenderer\nTHREE.PerspectiveCamera",
                    "expected_identifiers_missing": ["THREE.WebGLRenderer"],
                    "corrupt_identifier_tokens": ["THUEE.WebGLRenderer"],
                },
                {
                    "name": "constructor_unique_prefix",
                    "status": "pass",
                    "content": "new THREE.PerspectiveCamera(45, 1, 0.1, 1000)",
                    "expected_identifiers_missing": [],
                    "corrupt_identifier_tokens": [],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_installed_unique_prefix_identifier_probe"]
    assert quality["status"] == "open"
    assert probe["status"] == "review"
    assert probe["unique_prefix_failed"] is True
    assert probe["plain_list_failed"] is True
    assert probe["constructor_unique_prefix_passed"] is True
    assert probe["native_cache_pool_quant_enabled"] is True
    assert probe["generic_turboquant_kv_enabled"] is False
    assert probe["corrupt_identifiers_by_probe"]["list_plain"] == ["THIVE.WebGLRenderer"]
    assert probe["corrupt_identifiers_by_probe"]["list_unique_prefix"] == ["THUEE.WebGLRenderer"]


def test_objective_proof_digest_surfaces_dsv4_source_nocache_identifier_pass(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-list-nocache-source-20260523.json",
        {
            "status": "pass",
            "probe_elapsed_sec": 31.717,
            "health_before": {
                "native_cache": {
                    "prefix": False,
                    "paged": False,
                    "block_disk_l2": False,
                    "pool_quant": {"enabled": False, "env": "0"},
                    "generic_turboquant_kv": {"enabled": False},
                }
            },
            "probe": {
                "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTHREE.BoxGeometry\nTHREE.MeshBasicMaterial",
                "identifier_counts": {
                    "THREE.Scene": 1,
                    "THREE.WebGLRenderer": 1,
                    "THREE.PerspectiveCamera": 1,
                    "THREE.BoxGeometry": 1,
                    "THREE.MeshBasicMaterial": 1,
                },
                "usage": {"completion_tokens": 32},
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_source_nocache_identifier_probe"]
    assert quality["status"] == "open"
    assert probe["status"] == "pass"
    assert probe["all_identifiers_present"] is True
    assert probe["native_cache_prefix"] is False
    assert probe["native_cache_paged"] is False
    assert probe["native_cache_block_disk_l2"] is False
    assert probe["native_cache_pool_quant_enabled"] is False
    assert probe["generic_turboquant_kv_enabled"] is False
    assert probe["probe_elapsed_sec"] == 31.717


def test_objective_proof_digest_surfaces_dsv4_same_prompt_cache_boundary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-sameprompt-nocache-source-20260523.json",
        {
            "status": "fail",
            "health_before": {
                "native_cache": {
                    "prefix": False,
                    "paged": False,
                    "block_disk_l2": False,
                    "pool_quant": {"enabled": False, "env": "0"},
                    "generic_turboquant_kv": {"enabled": False},
                }
            },
            "probe": {
                "analysis": {
                    "content": "THREE.ScScene\nTHREE.WebWebGLRenderer\nTHREE.PerspectiveCamera\nTHREE.BBoxGeometry",
                    "missing_identifiers": [
                        "THREE.Scene",
                        "THREE.WebGLRenderer",
                        "THREE.BoxGeometry",
                    ],
                    "has_common_corruptions": True,
                },
                "usage": {"completion_tokens": 32},
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-cache-source-comparison-20260523.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "pooloff",
                    "status": "fail",
                    "artifact": "build/current-dsv4-live-identifier-cache-source-pooloff-20260523.json",
                    "failures": [
                        {
                            "label": "cold_cache_allowed",
                            "missing": ["THREE.Scene"],
                            "content": "THREE.ScScene",
                        }
                    ],
                },
                {
                    "name": "poolon",
                    "status": "fail",
                    "artifact": "build/current-dsv4-live-identifier-cache-source-poolon-20260523.json",
                    "failures": [
                        {
                            "label": "cold_cache_allowed",
                            "missing": ["THREE.Scene"],
                            "content": "THREE.ScScene",
                        }
                    ],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    boundary = quality["details"]["current_source_same_prompt_cache_boundary"]
    assert quality["status"] == "open"
    assert boundary["same_prompt_nocache_failed"] is True
    assert boundary["cache_enabled_pooloff_failed"] is True
    assert boundary["cache_enabled_poolon_failed"] is True
    assert boundary["pool_quant_is_not_differentiator"] is True
    assert boundary["cache_hit_restore_not_proven_by_short_prompt"] is True
    assert boundary["same_prompt_nocache"]["missing_identifiers"] == [
        "THREE.Scene",
        "THREE.WebGLRenderer",
        "THREE.BoxGeometry",
    ]


def test_objective_proof_digest_downgrades_pass_rows_with_missing_evidence(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "build/dev-ui-smoke-20260521/summary.json").unlink()

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Server default max output and max context are distinct and map to correct CLI flags"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == ["build/dev-ui-smoke-20260521/summary.json"]
    assert row["details"]["evidence_files_present"]["build/dev-ui-smoke-20260521/summary.json"] is False


def test_objective_proof_digest_rejects_no_cache_dsv4_multi_tool_proof(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop/result.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["cmd"] = [
        "python",
        "-m",
        "vmlx_engine.cli",
        "serve",
        "/models/dsv4",
        "--disable-prefix-cache",
        "--tool-call-parser",
        "dsml",
    ]
    proof["health"]["native_cache"]["prefix"] = False
    proof["health"]["native_cache"]["paged"] = False
    proof["health"]["native_cache"]["block_disk_l2"] = False
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    assert row["status"] == "open"
    assert row["details"]["launch_has_disable_prefix_cache"] is True
    assert row["details"]["native_cache_prefix"] is False


def test_objective_proof_digest_rejects_default_cache_multi_tool_without_cache_hit(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop/result.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    for item in proof["rounds"]:
        item["usage"] = {
            "input_tokens_details": {"cached_tokens": 0, "cache_detail": ""}
        }
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    assert row["status"] == "open"
    assert row["details"]["tool_loop_cached_tokens"] == 0
    assert row["details"]["tool_loop_cache_detail_has_dsv4"] is False


def test_objective_proof_digest_surfaces_skipped_default_cache_multi_tool_gate(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop/result.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.clear()
    proof.update(
        {
            "status": "skipped",
            "reason": "insufficient_free_memory",
            "cmd": ["python", "-m", "vmlx_engine.cli", "serve", "/models/dsv4"],
        }
    )
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    assert row["status"] == "open"
    assert row["details"]["artifact_status"] == "skipped"
    assert row["details"]["artifact_reason"] == "insufficient_free_memory"


def test_objective_proof_digest_marks_api_cache_contract_open_without_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "build/current-api-cache-contract-proof-20260521.json").unlink()

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Current-source API adapters and non-DSV4 cache contracts are no-heavy covered"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [
        "build/current-api-cache-contract-proof-20260521.json"
    ]


def test_objective_proof_digest_accepts_api_cache_contract_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Current-source API adapters and non-DSV4 cache contracts are no-heavy covered"]
    assert row["status"] == "pass"
    assert row["details"]["contract_checks"]["responses_sampling_kwargs"] is True
    assert row["details"]["contract_checks"]["dsv4_dsml_parser_residue_rejection"] is True
    assert row["details"]["missing_evidence"] == []
    assert row["details"]["stale_source_hashes"] == []


def test_objective_proof_digest_accepts_panel_settings_contract_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Panel settings keep DSV4 cache, max output, and max context controls unambiguous"]
    assert row["status"] == "pass"
    assert row["details"]["contract_checks"]["dsv4_explicit_prefix_off_disables_native_flags"] is True
    assert row["details"]["contract_checks"]["max_output_context_cli_split"] is True
    assert row["details"]["missing_evidence"] == []
    assert row["details"]["stale_source_hashes"] == []


def test_objective_proof_digest_requires_max_output_context_contract_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "build/current-max-output-context-contract-20260521.json").unlink()

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Server default max output and max context are distinct and map to correct CLI flags"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [
        "build/current-max-output-context-contract-20260524-after-strict-dsv4-canary.json"
    ]


def test_objective_proof_digest_accepts_max_output_context_contract_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Server default max output and max context are distinct and map to correct CLI flags"]
    assert row["status"] == "pass"
    assert row["details"]["contract_checks"]["request_output_caps_do_not_mutate_server_default"] is True
    assert row["details"]["contract_checks"]["new_chat_output_caps_are_not_inherited_or_made_sticky"] is True
    assert row["details"]["missing_evidence"] == []
    assert row["details"]["stale_source_hashes"] == []


def test_objective_proof_digest_requires_model_family_parser_artifact_contracts(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "build/current-parser-registry-contract-20260521.json").unlink(
        missing_ok=True
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["High-risk model family parser, artifact, and launch policy gates are current"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [
        "build/current-parser-registry-contract-20260521.json"
    ]


def test_objective_proof_digest_accepts_model_family_parser_artifact_contracts(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["High-risk model family parser, artifact, and launch policy gates are current"]
    assert row["status"] == "pass"
    assert row["details"]["model_family"]["missing_rows"] == []
    assert row["details"]["parser_registry"]["missing_markers"] == []
    assert row["details"]["model_artifact_format"]["missing_markers"] == []
    assert row["details"]["model_family"]["missing_source_hashes"] == []
    assert row["details"]["parser_registry"]["stale_source_hashes"] == []
    assert row["details"]["model_artifact_format"]["stale_source_hashes"] == []


def test_objective_proof_digest_requires_generation_mtp_vl_contracts(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "build/current-native-mtp-contract-20260521.json").unlink(
        missing_ok=True
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Generation defaults, Native MTP, and VL media gates are current"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [
        "build/current-native-mtp-contract-20260521.json"
    ]


def test_objective_proof_digest_accepts_generation_mtp_vl_contracts(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Generation defaults, Native MTP, and VL media gates are current"]
    assert row["status"] == "pass"
    assert row["details"]["generation_defaults"]["missing_markers"] == []
    assert row["details"]["native_mtp"]["missing_markers"] == []
    assert row["details"]["vl_media"]["missing_markers"] == []
    assert row["details"]["generation_defaults"]["stale_source_hashes"] == []
    assert row["details"]["native_mtp"]["stale_source_hashes"] == []
    assert row["details"]["vl_media"]["stale_source_hashes"] == []


def test_objective_proof_digest_surfaces_qwen_jang_packaged_speed_review(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import (
        QWEN_JANG_PACKAGED_SPEED_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    packaged = tmp_path / QWEN_JANG_PACKAGED_SPEED_REL
    payload = json.loads(packaged.read_text(encoding="utf-8"))
    payload["results"][0]["status"] = "review"
    payload["results"][0]["notes"] = [
        "PP below expected 600.00: 294.83, 288.07, 249.49"
    ]
    payload["results"][0]["runtime_wheels"] = {
        "mlx": ["cp312-cp312-macosx_14_0_arm64"],
        "mlx-metal": ["py3-none-macosx_14_0_arm64"],
    }
    payload["results"][0]["pp_rows"] = [
        {"target_tokens": 1024, "pp_wall_tok_s": 294.83, "loopish": False},
        {"target_tokens": 4096, "pp_wall_tok_s": 288.07, "loopish": False},
        {"target_tokens": 16384, "pp_wall_tok_s": 249.49, "loopish": False},
    ]
    packaged.write_text(json.dumps(payload), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen/JANG packaged MX matmul speed is release-cleared"]
    assert row["status"] == "open"
    assert row["details"]["packaged"]["status"] == "review"
    assert row["details"]["packaged"]["notes"] == [
        "PP below expected 600.00: 294.83, 288.07, 249.49"
    ]
    assert row["details"]["packaged"]["min_pp_wall_tok_s"] == 249.49


def test_objective_proof_digest_accepts_qwen_jang_speed_when_source_and_packaged_pass(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import (
        QWEN_JANG_PACKAGED_SPEED_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen/JANG packaged MX matmul speed is release-cleared"]
    assert row["status"] == "pass"
    assert row["details"]["source"]["status"] == "pass"
    assert row["details"]["packaged"]["status"] == "pass"
    assert row["details"]["source"]["min_pp_wall_tok_s"] >= 600
    assert row["details"]["packaged"]["min_pp_wall_tok_s"] >= 600
    assert "packaged-tahoe-dmg" in QWEN_JANG_PACKAGED_SPEED_REL


def test_objective_proof_digest_keeps_qwen_prefill_review_as_historical_diagnostic(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import (
        QWEN_NATIVE_MTP_PREFILL_SPEED_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    path = tmp_path / QWEN_NATIVE_MTP_PREFILL_SPEED_REL
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["status"] = "review"
    payload["results"][0]["notes"] = [
        "PP below expected 600.00: 143.74, 146.48, 146.66"
    ]
    payload["results"][0]["pp_rows"] = [
        {"target_tokens": 1024, "pp_wall_tok_s": 143.74, "loopish": False},
        {"target_tokens": 4096, "pp_wall_tok_s": 146.48, "loopish": False},
        {"target_tokens": 16384, "pp_wall_tok_s": 146.66, "loopish": False},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen 27B JANG_4M prompt-processing speed floor is release-cleared"]
    assert row["status"] == "pass"
    assert row["details"]["native_mtp_prefill_source_native_wheels"]["status"] == "review"
    assert row["details"]["native_mtp_prefill_source_native_wheels"]["min_pp_wall_tok_s"] == 143.74
    assert row["details"]["native_mtp_after_norm_shift_default_cache"]["status"] == "pass"
    assert row["details"]["text_loader"]["min_pp_wall_tok_s"] >= 600


def test_objective_proof_digest_opens_qwen_prompt_processing_when_norm_shift_clearance_is_review(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import (
        QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    path = tmp_path / QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["status"] = "review"
    payload["results"][0]["notes"] = ["loopish output detected"]
    payload["results"][0]["pp_rows"] = [
        {"target_tokens": 1024, "pp_wall_tok_s": 710.1, "loopish": True},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen 27B JANG_4M prompt-processing speed floor is release-cleared"]
    assert row["status"] == "open"
    assert row["details"]["native_mtp_after_norm_shift_default_cache"]["status"] == "review"
    assert row["details"]["native_mtp_after_norm_shift_default_cache"]["loopish_values"] == [True]


def test_objective_proof_digest_explains_qwen_mllm_prefill_trace_bottleneck(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import (
        QWEN_NATIVE_MTP_PREFILL_TRACE_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    path = tmp_path / QWEN_NATIVE_MTP_PREFILL_TRACE_REL
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["status"] = "review"
    payload["results"][0]["notes"] = ["PP below expected 600.00: 286.68, 285.13, 288.24"]
    payload["results"][0]["health_after"]["scheduler"]["batch_generator"]["last_prefill_trace"] = {
        "has_images": False,
        "is_hybrid": True,
        "native_mtp": True,
        "prefix_cache_enabled": True,
        "preprocess_ms": 0.085,
        "forward_ms": 10.967,
        "logits_eval_ms": 162.888,
        "sample_ms": 167.646,
        "total_ms": 179.031,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen 27B JANG_4M prompt-processing speed floor is release-cleared"]
    diagnosis = row["details"]["prefill_trace"]["diagnosis"]
    assert diagnosis["text_only_mllm_path"] is True
    assert diagnosis["preprocess_is_not_bottleneck"] is True
    assert diagnosis["logits_eval_dominates_forward"] is True
    assert diagnosis["sample_dominates_total"] is True
    assert diagnosis["suspected_bottleneck"] == "logits/sample materialization"


def test_objective_proof_digest_surfaces_qwen_no_prefix_logits_trial(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    mtp_prefill = tmp_path / "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json"
    mtp_payload = json.loads(mtp_prefill.read_text(encoding="utf-8"))
    mtp_payload["results"][0]["status"] = "review"
    mtp_payload["results"][0]["notes"] = ["PP below expected 600.00: 286.56"]
    mtp_payload["results"][0]["pp_rows"] = [
        {"pp_wall_tok_s": 286.56, "loopish": False},
    ]
    mtp_prefill.write_text(json.dumps(mtp_payload), encoding="utf-8")
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-no-prefix-logits-20260523.json",
        {
            "results": [
                {
                    "status": "review",
                    "notes": ["PP below expected 600.00: 271.38, 282.35, 256.75"],
                    "pp_rows": [
                        {"pp_wall_tok_s": 271.38, "loopish": False},
                        {"pp_wall_tok_s": 282.35, "loopish": False},
                        {"pp_wall_tok_s": 256.75, "loopish": False},
                    ],
                    "bundle_sampling": {"decode_tps_wall": 24.42, "loopish": False},
                }
            ]
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen 27B JANG_4M prompt-processing speed floor is release-cleared"]
    trial = row["details"]["native_mtp_no_prefix_logits_trial"]
    assert row["status"] == "pass"
    assert trial["status"] == "review"
    assert trial["min_pp_wall_tok_s"] == 256.75
    assert trial["clears_prompt_processing_floor"] is False


def test_objective_proof_digest_surfaces_qwen_vlm_route_ab_trials(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    mtp_prefill = tmp_path / "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json"
    mtp_payload = json.loads(mtp_prefill.read_text(encoding="utf-8"))
    mtp_payload["results"][0]["status"] = "review"
    mtp_payload["results"][0]["notes"] = ["PP below expected 600.00: 286.56"]
    mtp_payload["results"][0]["pp_rows"] = [
        {"pp_wall_tok_s": 286.56, "loopish": False},
    ]
    mtp_prefill.write_text(json.dumps(mtp_payload), encoding="utf-8")
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-hybrid-long-prefix-split-20260523.json",
        {
            "results": [
                {
                    "status": "review",
                    "notes": ["PP below expected 600.00: 288.29, 297.96, 284.58"],
                    "pp_rows": [
                        {"pp_wall_tok_s": 288.29, "loopish": False},
                        {"pp_wall_tok_s": 297.96, "loopish": False},
                        {"pp_wall_tok_s": 284.58, "loopish": False},
                    ],
                    "bundle_sampling": {"decode_tps_wall": 25.53, "loopish": False},
                }
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-kvnone-20260523.json",
        {
            "results": [
                {
                    "status": "review",
                    "notes": ["PP below expected 600.00: 274.78, 285.30, 277.18"],
                    "pp_rows": [
                        {"pp_wall_tok_s": 274.78, "loopish": False},
                        {"pp_wall_tok_s": 285.30, "loopish": False},
                        {"pp_wall_tok_s": 277.18, "loopish": False},
                    ],
                    "bundle_sampling": {"decode_tps_wall": 25.47, "loopish": False},
                }
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-decode-speed-live-qwen27-jang4m-mtp-route-trace-20260523.json",
        {
            "results": [
                {
                    "status": "review",
                    "notes": ["PP below expected 600.00: 274.09, 294.17, 268.92"],
                    "pp_rows": [
                        {"pp_wall_tok_s": 274.09, "loopish": False},
                        {"pp_wall_tok_s": 294.17, "loopish": False},
                        {"pp_wall_tok_s": 268.92, "loopish": False},
                    ],
                    "bundle_sampling": {"decode_tps_wall": 23.54, "loopish": False},
                    "health_after": {
                        "scheduler": {
                            "batch_generator": {
                                "last_prefill_trace": {
                                    "has_images": False,
                                    "is_hybrid": True,
                                    "native_mtp": True,
                                    "prefix_cache_enabled": True,
                                    "language_model_class": "mlx_vlm.models.qwen3_5.language.LanguageModel",
                                    "force_text_rope_1d": True,
                                    "supports_return_logits": True,
                                    "forward_ms": 169.813,
                                    "logits_eval_ms": 36.577,
                                    "total_ms": 211.097,
                                }
                            }
                        }
                    },
                }
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-qwen-forward-path-ab-1024-vlm-loader-20260523.json",
        {
            "comparison": [
                {
                    "prompt_tokens": 1024,
                    "text_pp_tok_s": 940.6765926229684,
                    "vlm_pp_tok_s": 301.6848317127802,
                    "text_over_vlm": 3.118077190962461,
                }
            ],
            "generation_defaults_policy": "raw-forward-only; no sampler or generation kwargs are accepted",
        },
    )
    _write_json(
        tmp_path,
        "build/current-qwen-forward-path-ab-4096-vlm-loader-20260523.json",
        {
            "comparison": [
                {
                    "prompt_tokens": 4096,
                    "text_pp_tok_s": 930.4369734441283,
                    "vlm_pp_tok_s": 296.7579107615906,
                    "text_over_vlm": 3.1353400859855185,
                }
            ],
            "generation_defaults_policy": "raw-forward-only; no sampler or generation kwargs are accepted",
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen 27B JANG_4M prompt-processing speed floor is release-cleared"]
    prefix_split = row["details"]["native_mtp_hybrid_long_prefix_split_trial"]
    kvnone = row["details"]["native_mtp_kvnone_trial"]
    route_trace = row["details"]["native_mtp_vlm_route_trace"]
    forward_ab = row["details"]["raw_forward_path_ab"]
    assert row["status"] == "pass"
    assert prefix_split["clears_prompt_processing_floor"] is False
    assert prefix_split["min_pp_wall_tok_s"] == 284.58
    assert kvnone["clears_prompt_processing_floor"] is False
    assert kvnone["min_pp_wall_tok_s"] == 274.78
    assert route_trace["last_prefill_trace"]["language_model_class"] == "mlx_vlm.models.qwen3_5.language.LanguageModel"
    assert route_trace["diagnosis"]["uses_vlm_language_copy"] is True
    assert route_trace["diagnosis"]["force_text_rope_1d"] is True
    assert route_trace["diagnosis"]["supports_return_logits"] is True
    assert forward_ab["raw_forward_only"] is True
    assert forward_ab["min_text_over_vlm"] == 3.118077190962461
    assert forward_ab["comparisons"][0]["prompt_tokens"] == 1024
    assert forward_ab["comparisons"][1]["prompt_tokens"] == 4096


def test_objective_proof_digest_accepts_qwen_native_mtp_decode_ab_when_equivalent_and_faster(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen native MTP live decode speed and output equivalence are release-cleared"]
    assert row["status"] == "pass"
    assert row["details"]["speedup_vs_baseline"] >= 1.1
    assert row["details"]["baseline_decode_tps_wall"] >= 20
    assert row["details"]["native_mtp_decode_tps_wall"] >= 25
    assert row["details"]["native_mtp_acceptance_rate"] >= 0.5


def test_objective_proof_digest_accepts_qwen_prompt_processing_when_text_and_mtp_prefill_pass(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Qwen 27B JANG_4M prompt-processing speed floor is release-cleared"]
    assert row["status"] == "pass"
    assert row["details"]["text_loader"]["min_pp_wall_tok_s"] >= 600
    assert row["details"]["native_mtp_after_norm_shift_default_cache"]["min_pp_wall_tok_s"] >= 600
    assert row["details"]["prefill_trace"]["last_prefill_trace"]["logits_eval_ms"] == 204.2


def test_objective_proof_digest_rejects_stale_panel_settings_contract_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx").write_text(
        "changed", encoding="utf-8"
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Panel settings keep DSV4 cache, max output, and max context controls unambiguous"]
    assert row["status"] == "open"
    assert row["details"]["stale_source_hashes"] == [
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx"
    ]


def test_objective_proof_digest_rejects_api_cache_contract_without_source_hashes(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    contract_path = tmp_path / "build/current-api-cache-contract-proof-20260521.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.pop("source_hashes")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Current-source API adapters and non-DSV4 cache contracts are no-heavy covered"]
    assert row["status"] == "open"
    assert row["details"]["missing_source_hashes"] == [
        "vmlx_engine/server.py",
        "vmlx_engine/tool_parsers/dsml_tool_parser.py",
        "tests/test_engine_audit.py",
        "tests/test_batching.py",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_tq_disk_cache.py",
        "tests/test_dsml_tool_parser.py",
        "tests/test_tool_format.py",
    ]


def test_objective_proof_digest_rejects_stale_api_cache_contract_hashes(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / "tests/test_engine_audit.py").write_text(
        "changed after proof\n",
        encoding="utf-8",
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Current-source API adapters and non-DSV4 cache contracts are no-heavy covered"]
    assert row["status"] == "open"
    assert row["details"]["stale_source_hashes"] == ["tests/test_engine_audit.py"]


def test_objective_proof_digest_accepts_dsv4_quality_clearance_artifact(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(tmp_path, "build/dsv4-source-identifier/result.json", {"ok": True})
    _write_json(tmp_path, "build/dsv4-source-full-output/result.json", {"ok": True})
    _write_json(
        tmp_path,
        "build/current-dsv4-identifier-count-ablation-20260521/result.json",
        {"ok": True},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-list-nocache-source-20260523.json",
        {
            "status": "pass",
            "health_before": {
                "native_cache": {
                    "prefix": False,
                    "paged": False,
                    "block_disk_l2": False,
                    "pool_quant": {"enabled": False, "env": "0"},
                    "generic_turboquant_kv": {"enabled": False},
                }
            },
            "probe": {
                "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera",
                "identifier_counts": {
                    "THREE.Scene": 1,
                    "THREE.WebGLRenderer": 1,
                    "THREE.PerspectiveCamera": 1,
                },
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-sameprompt-nocache-source-20260523.json",
        {
            "status": "pass",
            "probe": {
                "analysis": {
                    "content": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTHREE.BoxGeometry",
                    "missing_identifiers": [],
                    "has_common_corruptions": False,
                },
                "usage": {"completion_tokens": 32},
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-cache-source-comparison-20260523.json",
        {
            "status": "pass",
            "cases": [
                {
                    "name": "pooloff",
                    "status": "pass",
                    "artifact": "build/current-dsv4-live-identifier-cache-source-pooloff-20260523.json",
                    "failures": [],
                },
                {
                    "name": "poolon",
                    "status": "pass",
                    "artifact": "build/current-dsv4-live-identifier-cache-source-poolon-20260523.json",
                    "failures": [],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "docs/internal/release-gates/20260520_sisyphus_dsv4_identifier_gate_jang_affine_current/result.json",
        {"ok": True},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-thinking-on-responses-controls.json",
        {
            "status": "pass",
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "prompt_endswith_assistant_think_close": True,
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260524-thinking-on-responses-controls.json",
        {
            "status": "dry_run",
            "case_count": 1,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "request_overrides": {"enable_thinking": False, "max_tokens": 512},
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-long-output-quality-clearance-20260521.json",
        {
            "status": "pass",
            "checks": {
                "identifier_integrity": True,
                "threejs_single_file": True,
                "no_markdown_fence": True,
                "no_corrupt_identifiers": True,
                "non_length_stop": True,
                "source_or_rebuilt_body_clearance": True,
                "cached_vs_no_cache_semantic_equivalence": True,
            },
            "artifacts": {
                "identifier_gate": "build/dsv4-source-identifier/result.json",
                "full_output_gate": "build/dsv4-source-full-output/result.json",
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    assert quality["status"] == "pass"
    assert quality["details"]["clearance_checks"]["identifier_integrity"] is True
    assert sum(item["status"] == "open" for item in digest["requirements"]) == 0


def test_objective_proof_digest_rejects_quality_clearance_with_missing_artifacts(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-long-output-quality-clearance-20260521.json",
        {
            "status": "pass",
            "checks": {
                "identifier_integrity": True,
                "threejs_single_file": True,
                "no_markdown_fence": True,
                "no_corrupt_identifiers": True,
                "non_length_stop": True,
                "source_or_rebuilt_body_clearance": True,
                "cached_vs_no_cache_semantic_equivalence": True,
            },
            "artifacts": {
                "identifier_gate": "build/dsv4-source-identifier/result.json",
                "full_output_gate": "build/dsv4-source-full-output/result.json",
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    assert quality["status"] == "open"
    assert quality["details"]["clearance_artifact_paths_present"] == {
        "identifier_gate": False,
        "full_output_gate": False,
    }
    assert quality["details"]["missing_clearance_artifacts"] == [
        "build/dsv4-source-identifier/result.json",
        "build/dsv4-source-full-output/result.json",
    ]

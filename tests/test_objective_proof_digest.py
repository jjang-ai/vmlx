import json
from pathlib import Path


def _write_json(root: Path, rel: str, obj: dict) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_objective_proof_digest_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import summarize_objective_proof as objective

    assert objective.DEFAULT_OUT == Path(
        "build/current-objective-proof-audit-20260531-step37-reasoning-ledger.json"
    )


def test_objective_proof_digest_live_smoke_pointers_match_release_manifest_current_map():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import summarize_objective_proof as objective

    assert objective.ALL_LOCAL_MODEL_SMOKE_ZAYA_VL_JANGTQ4_REL == (
        manifest.CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["zaya_vl_jangtq4"]
    )
    assert objective.ALL_LOCAL_MODEL_SMOKE_LING_BAILING_JANGTQ_REL == (
        manifest.CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["ling_flash_tq"]
    )
    assert objective.ALL_LOCAL_MODEL_SMOKE_QWEN36_MXFP4_CRACK_REL == (
        manifest.CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    )
    assert objective.ALL_LOCAL_MODEL_SMOKE_HY3_JANGTQ2_REL == (
        manifest.CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["hy3_preview_jangtq2"]
    )
    assert objective.ALL_LOCAL_MODEL_SMOKE_MINIMAX_SMALL_JANGTQ_REL == (
        manifest.CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["minimax_m27_tq_k"]
    )


def test_objective_proof_digest_release_manifest_pointer_matches_current_suite():
    from tests.cross_matrix import run_current_regression_suite as suite
    from tests.cross_matrix import summarize_objective_proof as objective

    cmd = suite.CURRENT_SUITE_COMMANDS["release_regression_manifest"]

    assert "--out" in cmd
    assert objective.CURRENT_RELEASE_REGRESSION_MANIFEST_REL == cmd[
        cmd.index("--out") + 1
    ]


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
                "--reasoning-parser",
                "deepseek_r1",
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
                    "executed_tools": [{"name": "write_file"}],
                    "usage": {
                        "input_tokens_details": {
                            "cached_tokens": 192,
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
            ],
            "checks": {
                "tool_sequence_ordered": True,
                "final_done": True,
                "file_written": True,
                "code_file_written_exact": True,
                "native_cache": True,
                "native_prefix": True,
                "native_paged": True,
                "native_l2": True,
                "generic_tq_kv_off": True,
                "cached_tokens_seen": True,
                "dsv4_cache_detail_seen": True,
            },
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
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json",
        {
            "status": "pass",
            "row": "gemma4_26b_jang4m",
            "request_route": "responses",
            "max_tokens": 512,
            "response_contract": {"expect_visible_content": True},
            "results": [
                {
                    "status": "ok",
                    "response_rails": {
                        "visible_chars": 517,
                        "reasoning_chars": 1056,
                    },
                }
            ],
        },
    )
    for rel in (
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-contract-20260524.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-nocache-20260524.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-512-nocache-20260524.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingoff-visible-nocache-20260524.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingbudget16-visible-nocache-20260524.json",
    ):
        _write_json(
            tmp_path,
            rel,
            {
                "status": "pass",
                "request_route": "responses",
                "results": [
                    {
                        "status": "ok",
                        "response_rails": {
                            "visible_chars": 1,
                            "reasoning_chars": 1,
                        },
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-local-generation-metadata-audit-20260524-gemma4-visible-budget.json",
        {"status": "pass", "review_notes": {}},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-live-identifier-matrix-20260523.json",
        {
            "status": "pass",
            "probes": [
                {
                    "id": "copy_identifiers_only",
                    "status": "probe_failed",
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
    from tests.cross_matrix.summarize_objective_proof import API_CACHE_CONTRACT_REL

    _write_json(
        tmp_path,
        API_CACHE_CONTRACT_REL,
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
    from tests.cross_matrix.summarize_objective_proof import PANEL_SETTINGS_CONTRACT_REL

    _write_json(
        tmp_path,
        PANEL_SETTINGS_CONTRACT_REL,
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
        MAX_OUTPUT_CONTEXT_CONTRACT_REL,
        MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS,
        MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES,
    )

    for rel in MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES:
        if not (tmp_path / rel).exists():
            _write_json(tmp_path, rel, {"fixture": rel})
    _write_json(
        tmp_path,
        MAX_OUTPUT_CONTEXT_CONTRACT_REL,
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
        MODEL_ARTIFACT_FORMAT_CONTRACT_REL,
        MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS,
        MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
        MODEL_FAMILY_CONTRACT_REL,
        MODEL_FAMILY_CONTRACT_CHECKS,
        MODEL_FAMILY_SOURCE_HASH_FILES,
        PARSER_REGISTRY_CONTRACT_REL,
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
        MODEL_FAMILY_CONTRACT_REL,
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
        PARSER_REGISTRY_CONTRACT_REL,
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
        MODEL_ARTIFACT_FORMAT_CONTRACT_REL,
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
        GENERATION_DEFAULTS_CONTRACT_REL,
        GENERATION_DEFAULTS_CONTRACT_CHECKS,
        REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX,
        GENERATION_DEFAULTS_SOURCE_HASH_FILES,
        NATIVE_MTP_CONTRACT_REL,
        NATIVE_MTP_CONTRACT_CHECKS,
        NATIVE_MTP_SOURCE_HASH_FILES,
        VL_MEDIA_CONTRACT_REL,
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
        GENERATION_DEFAULTS_CONTRACT_REL,
        {
            "status": "pass",
            "checks": {
                key: True
                for key in GENERATION_DEFAULTS_CONTRACT_CHECKS
            },
            "failed": [],
            "missing_markers": [],
            "generation_defaults_family_matrix": {
                row: {
                    "status": "pass",
                    "checks": {},
                    "missing_markers": [],
                }
                for row in REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX
            },
            "source_hashes": {
                rel: _sha256(tmp_path / rel)
                for rel in GENERATION_DEFAULTS_SOURCE_HASH_FILES
            },
        },
    )
    _write_json(
        tmp_path,
        NATIVE_MTP_CONTRACT_REL,
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
        VL_MEDIA_CONTRACT_REL,
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
    assert rows["Ling/Bailing multilingual output quality is release-cleared"][
        "status"
    ] == "open"
    open_requirements = [
        item["requirement"] for item in digest["requirements"] if item["status"] == "open"
    ]
    assert open_requirements == [
        "Ling/Bailing multilingual output quality is release-cleared",
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared",
        "Cross-family live multi-turn smoke matrix is release-cleared",
        "Real Electron UI cross-family live model matrix is release-cleared",
        "DSV4 long-output/code/file-generation quality is release-cleared",
    ]


def test_objective_proof_digest_tracks_ling_multilingual_cjk_leakage(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-strict-russian-nocache-bundled-4850c9c2-20260524.json",
        {
            "status": "pass",
            "requests": [
                {
                    "content": "1. 猎人奔跑在森林中追逐目标。",
                    "counts": {"cjk_chars": 17, "cyrillic_chars": 0, "latin_chars": 0},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-mxfp4-crack-strict-russian-nocache-bundled-4850c9c2-20260524.json",
        {
            "status": "fail",
            "requests": [
                {
                    "content": "- Кабаны появляются как враги; запас弹匣结束。",
                    "counts": {"cjk_chars": 4, "cyrillic_chars": 31, "latin_chars": 0},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-russian-prompt-variant-probe-20260524.json",
        {
            "ready": True,
            "rows": [
                {
                    "name": "exact_game_prompt",
                    "content": "4.碰撞检测实现目标击中与",
                    "counts": {"cjk_chars": 10, "cyrillic_chars": 12},
                },
                {
                    "name": "simple_russian_game_no_html",
                    "content": "Игрок выбирает роль охотника.",
                    "counts": {"cjk_chars": 0, "cyrillic_chars": 28},
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-server-repeat-russian-source-prefill-stream-20260524.json",
        {
            "ready": True,
            "rows": [
                {
                    "mode": "skip_cache",
                    "content": "Сцена создаётся с травой и деревьями.",
                    "counts": {"cjk_chars": 0, "cyrillic_chars": 32},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-server-repeat-russian-bundled-prefill-stream-20260524.json",
        {
            "ready": True,
            "rows": [
                {
                    "mode": "skip_cache",
                    "content": "4.碰撞检测实现目标击中与",
                    "counts": {"cjk_chars": 10, "cyrillic_chars": 0},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-server-repeat-russian-bundled-native-prefill-stream-20260524.json",
        {
            "ready": True,
            "rows": [
                {
                    "mode": "skip_cache",
                    "content": "Система оценивает очки и выживание.",
                    "counts": {"cjk_chars": 0, "cyrillic_chars": 31},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-production-family-live-ling-bundled-native-rerun-20260524.json",
        {
            "rows": [
                {
                    "live": {
                        "status": "FAIL",
                        "requests": [
                            {
                                "name": "ling_multilingual_loop_trigger",
                                "content": "4.碰撞检测实时触发",
                                "quality": {"cjk_chars": 8, "cyrillic_chars": 12},
                            }
                        ],
                    }
                }
            ]
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-cold-skipcache-repeat-bundled-native-20260524.json",
        {
            "ready": True,
            "rows": [
                {
                    "iteration": 1,
                    "content": "5.计分系统与关卡递增",
                    "counts": {"cjk_chars": 9, "cyrillic_chars": 10},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-ling-jangtq-batchgen-temp0-repeat-bundled-native-20260524.json",
        {
            "ready": True,
            "rows": [
                {
                    "iteration": 1,
                    "content": "4.碰撞检测实现目标",
                    "counts": {"cjk_chars": 8, "cyrillic_chars": 10},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "selected_cases": ["chat_off", "responses_off", "legacy_completion_raw"],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
        {
            "status": "pass",
            "reason": "sufficient_free_memory",
            "required_available_gb": 120.0,
            "model": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
            "selected_cases": [
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
            ],
            "case_count": 14,
            "telemetry": [
                {
                    "name": "preflight",
                    "system_memory": {"available_gb": 192.0, "total_gb": 256.0},
                }
            ],
        },
    )
    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    ling = rows["Ling/Bailing multilingual output quality is release-cleared"]
    assert ling["status"] == "open"
    assert ling["details"]["max_cjk_chars"] == 17
    assert ling["details"]["artifacts_with_cjk"] == [
        "build/current-ling-jangtq-strict-russian-nocache-bundled-4850c9c2-20260524.json",
        "build/current-ling-mxfp4-crack-strict-russian-nocache-bundled-4850c9c2-20260524.json",
        "build/current-ling-jangtq-russian-prompt-variant-probe-20260524.json",
        "build/current-ling-jangtq-server-repeat-russian-bundled-prefill-stream-20260524.json",
        "build/current-production-family-live-ling-bundled-native-rerun-20260524.json",
        "build/current-ling-jangtq-cold-skipcache-repeat-bundled-native-20260524.json",
        "build/current-ling-jangtq-batchgen-temp0-repeat-bundled-native-20260524.json",
    ]


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
        "build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json",
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
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                        "prompt_endswith_assistant_think_open": True,
                        "prompt_endswith_assistant_think_close": False,
                        "prompt_tail": "renderer.render(scene, camera);<｜Assistant｜><think>",
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
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                        "prompt_endswith_assistant_think_open": True,
                        "prompt_endswith_assistant_think_close": False,
                    },
                },
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": True,
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
                    "name": "chat_max",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "has_markdown_fence": True,
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "prompt_endswith_assistant_think_open": False,
                        "prompt_endswith_assistant_think_close": True,
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
            "case_count": 10,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "request_overrides": {"enable_thinking": False, "max_tokens": 512},
                    "prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                    "requested_prompt_diagnostics": {"assistant_suffix_kind": "thinking_closed"},
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
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
                    "name": "chat_max",
                    "route": "chat",
                    "request_overrides": {
                        "enable_thinking": False,
                        "max_tokens": 512,
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
    assert probe["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json"
    )
    assert probe["failed_thinking_closed_cases"] == [
        "chat_max",
        "responses_off",
        "responses_off_rep1",
    ]
    assert probe["requested_thinking_closed_but_effective_open_cases"] == [
        "chat_off",
        "chat_off_rep1",
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
    assert probe["dry_run_case_count"] == 10
    assert probe["dry_run_suffixes"]["chat_max"] == "thinking_closed"
    assert probe["dry_run_requested_suffixes"]["chat_off"] == "thinking_closed"
    assert probe["dry_run_effective_suffixes"]["chat_off"] == "thinking_open"
    assert probe["dry_run_suffixes"]["legacy_completion_raw"] == "none"
    assert probe["dry_run_suffixes"]["responses_on"] == "thinking_open"
    assert probe["case_summaries"][0]["assistant_suffix_kind"] == "thinking_open"
    assert probe["case_summaries"][0]["requested_assistant_suffix_kind"] == "thinking_closed"
    assert probe["case_summaries"][0]["dsv4_policy_reason"] == "dsv4_direct_rail_identifier_unsafe"
    assert probe["case_summaries"][2]["normalized_exact"] is True
    assert probe["case_summaries"][0]["prompt_tail"].endswith("<think>")


def test_objective_proof_digest_prefers_latest_source_route_exactness_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-thinking-on-responses-controls.json",
        {
            "status": "pass",
            "cases": [
                {
                    "name": "legacy_stale_pass",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open"
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chat_max",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": [
                        "THREE.PerspectiveCamera",
                        "THREE.BoxGeometry",
                    ],
                    "corrupt_patterns": [
                        "BBoxGeometry",
                        "MMeshBasicMaterial",
                    ],
                    "has_markdown_fence": False,
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed"
                    },
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                        "prompt_tail": "No explanation, no markdown.<｜Assistant｜><think>",
                        "prompt_endswith_assistant_think_open": True,
                    },
                }
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_installed_prompt_rail_exactness_probe"]

    assert quality["status"] == "open"
    assert probe["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json"
    )
    assert probe["status"] == "fail"
    assert probe["failed_cases"] == ["chat_max"]
    assert probe["case_summaries"][0]["assistant_suffix_kind"] == "thinking_open"
    assert probe["case_summaries"][0]["requested_assistant_suffix_kind"] == "thinking_closed"


def test_objective_proof_digest_prefers_explicit_off_route_exactness_subset(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json",
        {
            "status": "pass",
            "cases": [
                {
                    "name": "old_stale_pass",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open"
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off",
                "responses_off",
                "legacy_completion_raw",
            ],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "token_count": 90,
                            "tail_ids": [128804, 128822],
                        },
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.MeshBasicMaterial"],
                    "corrupt_patterns": ["Three.MeshBasicMaterial"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "token_count": 90,
                            "tail_ids": [128804, 128822],
                        },
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_not_requested",
                    },
                },
            ],
        },
    )
    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_installed_prompt_rail_exactness_probe"]

    assert quality["status"] == "open"
    assert probe["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json"
    )
    assert probe["status"] == "fail"
    assert probe["failed_thinking_closed_cases"] == [
        "chat_off",
        "responses_off",
        "legacy_completion_raw",
    ]
    assert {case["dsv4_policy_reason"] for case in probe["case_summaries"]} == {
        "thinking_disabled",
        "thinking_not_requested",
    }


def test_objective_proof_digest_surfaces_current_dsv4_rail_boundary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off",
                "responses_off",
                "legacy_completion_raw",
            ],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "finish": "stop",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "token_count": 90,
                            "tail_ids": [128804, 128822],
                        },
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "finish": "completed",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "token_count": 90,
                            "tail_ids": [128804, 128822],
                        },
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "finish": "stop",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_not_requested",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "finish": "stop",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "finish": "completed",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off_rep1",
                "chat_on_rep1",
                "responses_off_rep1",
                "responses_on_rep1",
            ],
            "case_count": 4,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "finish": "stop",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "finish": "stop",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "finish": "completed",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "finish": "completed",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
            ],
        },
    )
    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]

    direct = quality["details"]["current_generated_only_direct_rail_exactness_subset"]
    thinking = quality["details"]["current_requested_thinking_exactness_subset"]
    rep1 = quality["details"][
        "current_rep1_direct_vs_requested_thinking_exactness_subset"
    ]
    assert quality["status"] == "open"
    assert direct["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json"
    )
    assert direct["status"] == "fail"
    assert direct["failed_thinking_closed_cases"] == [
        "chat_off",
        "responses_off",
        "legacy_completion_raw",
    ]
    assert {
        case["dsv4_policy_reason"] for case in direct["case_summaries"]
    } == {"thinking_disabled", "thinking_not_requested"}
    assert {
        tuple(case["corrupt_patterns"]) for case in direct["case_summaries"]
    } == {("WebWebGLRenderer",)}
    assert thinking["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json"
    )
    assert thinking["status"] == "pass"
    assert thinking["failed_cases"] == []
    assert thinking["thinking_open_cases_without_identifier_corruption"] == [
        "chat_on",
        "responses_on",
    ]
    assert {case["dsv4_policy_reason"] for case in thinking["case_summaries"]} == {
        "requested_thinking"
    }
    assert rep1["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json"
    )
    assert rep1["status"] == "fail"
    assert rep1["failed_thinking_closed_cases"] == [
        "chat_off_rep1",
        "responses_off_rep1",
    ]
    assert rep1["thinking_open_cases_without_identifier_corruption"] == [
        "chat_on_rep1",
        "responses_on_rep1",
    ]


def test_objective_proof_digest_surfaces_current_dsv4_webgl_logit_boundary(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json",
        {
            "schema": "vmlx-dsv4-direct-vs-thinking-webgl-logit-probe-v1",
            "cases": [
                {
                    "rail": "direct_off",
                    "prefix_name": "after_const_renderer_web",
                    "ranks": {"GL": 1, "Web": 5},
                    "top": [{"token": "GL"}, {"token": "3"}],
                },
                {
                    "rail": "direct_off",
                    "prefix_name": "after_const_renderer_three_dot",
                    "ranks": {"Web": 4},
                    "top": [
                        {"token": "<｜end▁of▁sentence｜>"},
                        {"token": "Three"},
                        {"token": " Web"},
                    ],
                },
                {
                    "rail": "direct_off",
                    "prefix_name": "after_camera_p",
                    "ranks": {"ers": 1, "Pers": 2},
                    "top": [{"token": "ers"}, {"token": "Pers"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_const_renderer_web",
                    "ranks": {"GL": 1, "Web": 5},
                    "top": [{"token": "GL"}, {"token": "Script"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_const_renderer_three_dot",
                    "ranks": {"Web": 1},
                    "top": [{"token": "Web"}, {"token": "<｜end▁of▁sentence｜>"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_camera_p",
                    "ranks": {"Pers": 1, "ers": 2},
                    "top": [{"token": "Pers"}, {"token": "ers"}],
                },
            ],
        },
    )
    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_direct_vs_thinking_webgl_logit_probe"]

    assert quality["status"] == "open"
    assert probe["artifact"].endswith(
        "current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json"
    )
    assert probe["direct_after_web_gl_rank"] == 1
    assert probe["thinking_after_web_gl_rank"] == 1
    assert probe["webweb_not_explained_by_after_web_rank"] is True
    assert probe["direct_after_three_dot_web_rank"] == 4
    assert probe["thinking_after_three_dot_web_rank"] == 1
    assert probe["direct_after_camera_p_top_token"] == "ers"
    assert probe["thinking_after_camera_p_top_token"] == "Pers"
    assert probe["direct_camera_p_prefers_suffix_over_whole_token"] is True


def test_objective_proof_digest_surfaces_dsv4_chatmax_prompt_trigger_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "canonical_return_fences",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": False,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "content": "const scene = new THREE.Scene();\nconst renderer = new THREE.WebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                        "prompt_tail": "code and no markdown fences:<｜Assistant｜><think>",
                    },
                },
                {
                    "name": "chatmax_original",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": [
                        "THREE.PerspectiveCamera",
                        "THREE.BoxGeometry",
                    ],
                    "corrupt_patterns": [
                        "BBoxGeometry",
                        "MMeshBasicMaterial",
                    ],
                    "has_markdown_fence": False,
                    "prompt_tokens": 97,
                    "completion_tokens": 445,
                    "content": "const camera = new THREE.PersPerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                        "prompt_tail": "No explanation, no markdown.<｜Assistant｜><think>",
                    },
                },
                {
                    "name": "copy_no_preserve",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.Scene"],
                    "corrupt_patterns": [],
                    "has_markdown_fence": False,
                    "prompt_tokens": 92,
                    "completion_tokens": 512,
                    "content": "",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": [
                "chat_off",
                "responses_off",
                "legacy_completion_raw",
            ],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_not_requested",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": [
                "chat_off",
                "responses_off",
                "legacy_completion_raw",
            ],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_not_requested",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json",
        {
            "schema": "vmlx-dsv4-cache-vs-full-logit-isolation-v1",
            "checks": [
                {
                    "name": "isolated_after_THREE_dot_P",
                    "full_top": [
                        {"token": "ers", "logprob": -1.001},
                        {"token": "ert", "logprob": -1.676},
                    ],
                    "incremental_top": [
                        {"token": "ers", "logprob": -1.380},
                        {"token": "ert", "logprob": -1.460},
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-logit-trace-v1",
            "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
            "events": [
                {
                    "generated_so_far_text": "THREE.P",
                    "top": [
                        {"token": "ers", "logprob": -1.001},
                        {"token": "ert", "logprob": -1.676},
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-warmup-ablation-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-warmup-ablation-v1",
            "runs": [
                {
                    "label": "normal_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                },
                {
                    "label": "skip_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
        {
            "status": "pass",
            "reason": "sufficient_free_memory",
            "required_available_gb": 120.0,
            "model": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
            "selected_cases": ["chat_off_rep1", "chat_on_rep1"],
            "case_count": 2,
            "telemetry": [
                {
                    "name": "preflight",
                    "system_memory": {"available_gb": 192.0, "total_gb": 256.0},
                }
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    trigger = quality["details"]["current_chatmax_prompt_trigger_probe"]

    assert quality["status"] == "open"
    assert trigger["artifact"].endswith(
        "current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json"
    )
    assert trigger["status"] == "fail"
    assert trigger["canonical_exact_cases"] == ["canonical_return_fences"]
    assert trigger["failed_cases"] == ["chatmax_original", "copy_no_preserve"]
    assert trigger["same_effective_rail"] is True
    assert trigger["all_effective_policy_reasons"] == [
        "dsv4_direct_rail_identifier_unsafe"
    ]
    assert trigger["blank_visible_cases"] == ["copy_no_preserve"]
    assert trigger["corrupt_patterns_by_case"]["chatmax_original"] == [
        "BBoxGeometry",
        "MMeshBasicMaterial",
    ]


def test_objective_proof_digest_surfaces_current_source_dsv4_rep1_ab(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-thinking-ab-prefill-logits-eval-20260525.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off_rep1",
                "chat_on_rep1",
                "responses_off_rep1",
                "responses_on_rep1",
            ],
            "case_count": 4,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json",
        {
            "status": "dry_run",
            "selected_cases": [
                "chat_off_rep1",
                "chat_on_rep1",
                "responses_off_rep1",
                "responses_on_rep1",
            ],
            "case_count": 4,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
                {
                    "name": "chat_on_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "request_overrides": {"enable_thinking": True},
                },
                {
                    "name": "responses_off_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
                {
                    "name": "responses_on_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "request_overrides": {"enable_thinking": True},
                },
            ],
        },
    )
    for rel in (
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json",
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json",
    ):
        _write_json(
            tmp_path,
            rel,
            {
                "status": "dry_run",
                "selected_cases": [
                    "chat_off_rep1",
                    "chat_off_no_punct_rep1",
                    "responses_off_rep1",
                    "responses_off_no_punct_rep1",
                ],
                "case_count": 4,
                "cases": [],
            },
        )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    current_source = quality["details"][
        "current_source_rep1_direct_vs_requested_thinking_exactness_subset"
    ]
    assert quality["status"] == "open"
    assert current_source["status"] == "fail"
    assert current_source["dry_run_artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json"
    )
    assert current_source["dry_run_present"] is True
    assert current_source["dry_run_status"] == "dry_run"
    assert current_source["dry_run_effective_suffixes"]["chat_off_rep1"] == "thinking_closed"
    assert current_source["dry_run_effective_suffixes"]["chat_on_rep1"] == "thinking_open"
    assert current_source["dry_run_requested_suffixes"]["responses_off_rep1"] == "thinking_closed"
    assert current_source["dry_run_request_overrides"]["chat_on_rep1"]["enable_thinking"] is True
    assert current_source["failed_cases"] == ["chat_off_rep1", "responses_off_rep1"]
    assert current_source["thinking_open_cases_without_identifier_corruption"] == [
        "chat_on_rep1",
        "responses_on_rep1",
    ]
    assert current_source["rep1_still_corrupt_patterns"] == {
        "chat_off_rep1": ["WebWebGLRenderer"],
        "responses_off_rep1": ["WebWebGLRenderer"],
    }


def test_objective_proof_digest_surfaces_current_source_dsv4_token_tail_ab(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-token-tail-source-ab-20260525.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off_rep1",
                "chat_on_rep1",
                "responses_off_rep1",
                "responses_on_rep1",
            ],
            "case_count": 4,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "tail_ids": [80547, 80547],
                            "token_count": 80,
                        },
                    },
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                        "token_diagnostics": {
                            "tail_ids": [19385, 80547],
                            "token_count": 79,
                        },
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "tail_ids": [80547, 80547],
                            "token_count": 80,
                        },
                    },
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                        "token_diagnostics": {
                            "tail_ids": [19385, 80547],
                            "token_count": 79,
                        },
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "selected_cases": ["chat_off", "responses_off", "legacy_completion_raw"],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "selected_cases": ["chat_off", "responses_off", "legacy_completion_raw"],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    token_tail = quality["details"][
        "current_source_token_tail_direct_vs_requested_thinking_exactness_subset"
    ]
    assert quality["status"] == "open"
    assert token_tail["artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-token-tail-source-ab-20260525.json"
    )
    assert token_tail["status"] == "fail"
    assert token_tail["failed_cases"] == ["chat_off_rep1", "responses_off_rep1"]
    assert token_tail["thinking_open_cases_without_identifier_corruption"] == [
        "chat_on_rep1",
        "responses_on_rep1",
    ]
    assert token_tail["case_summaries"][0]["token_tail_ids"] == [80547, 80547]
    assert token_tail["case_summaries"][0]["token_count"] == 80


def test_objective_proof_digest_summarizes_dsv4_direct_off_exactness_boundary(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-token-tail-source-ab-20260525.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off_rep1",
                "chat_on_rep1",
                "responses_off_rep1",
                "responses_on_rep1",
            ],
            "case_count": 4,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "requested_thinking",
                    },
                },
            ],
        },
    )

    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "selected_cases": ["chat_off", "responses_off", "legacy_completion_raw"],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        {
            "status": "pass",
            "selected_cases": ["chat_on", "responses_on"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_on",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
                {
                    "name": "responses_on",
                    "route": "responses",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
            ],
        },
    )

    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json",
        {
            "status": "fail",
            "selected_cases": ["chat_off_rep1", "responses_off_rep1"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    summary = quality["details"]["direct_off_exactness_boundary"]

    assert quality["status"] == "open"
    assert summary["direct_off_failing_cases"] == [
        "chat_off_rep1",
        "responses_off_rep1",
    ]
    assert summary["requested_thinking_exact_cases"] == [
        "chat_on_rep1",
        "responses_on_rep1",
    ]
    assert summary["direct_off_suffix_kind"] == "thinking_closed"
    assert summary["requested_thinking_suffix_kind"] == "thinking_open"
    assert summary["all_direct_off_failed_routes"] == [
        "chat",
        "completion",
        "responses",
    ]
    assert summary["direct_off_failure_spans_chat_responses_and_completion"] is True
    assert summary["requested_thinking_exact_routes"] == ["chat", "responses"]
    assert summary["requested_thinking_exact_spans_chat_and_responses"] is True
    assert "legacy_completion_raw" in summary["all_direct_off_failed_cases"]
    assert "build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json" in summary["direct_off_failure_artifacts"]
    assert "chat_on" in summary["all_requested_thinking_exact_cases"]
    assert summary["hidden_force_on_would_be_false_clearance"] is True
    assert summary["direct_off_release_blocker_active"] is True
    assert summary["requested_thinking_success_clears_direct_off"] is False
    assert summary["requested_thinking_success_is_diagnostic_only"] is True
    assert summary["root_boundary"] == (
        "Direct/off DSV4 exact-code generation fails under thinking_closed while "
        "requested-thinking exactness passes under thinking_open; hidden force-on "
        "does not clear explicit direct/off quality."
    )


def test_objective_proof_digest_summarizes_dsv4_exact_code_root_boundary(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-template-parity-diagnostic-20260524-1343.json",
        {
            "schema": "vmlx-dsv4-template-parity-diagnostic-v1",
            "all_sidecar_equals_tokenizer_template": True,
            "cases": [
                {
                    "name": "period_exact_code",
                    "enable_thinking": False,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_closed",
                    "boundary_tokens_tail": [".Ċ", "const"],
                },
                {
                    "name": "colon_exact_code",
                    "enable_thinking": False,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_closed",
                    "boundary_tokens_tail": [":Ċ", "const"],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json",
        {
            "schema": "vmlx-dsv4-hidden-reasoning-control-live-v1",
            "status": "fail",
            "cases": [
                {
                    "name": "period_no_reasoning_code_draft_system",
                    "exact": False,
                    "hidden_contains_corruption": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                },
                {
                    "name": "period_verify_identifiers_system",
                    "exact": False,
                    "hidden_contains_corruption": True,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-prefill-execution-variant-logits-20260524.json",
        {
            "schema": "vmlx-dsv4-prefill-execution-variant-logits-v1",
            "variants": [
                {"name": "plain", "top": [{"token": "ers"}, {"token": "Pers"}]},
                {"name": "stream", "top": [{"token": "ers"}, {"token": "Pers"}]},
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-prompt-variant-logit-probe-20260524.json",
        {
            "schema": "vmlx-dsv4-prompt-variant-logit-probe-v1",
            "cases": [
                {
                    "name": "original_list_fourth_after_TH",
                    "candidates": [
                        {"label": "correct_REE", "first_token_logprob": -0.7},
                        {"label": "corrupt_IVE", "first_token_logprob": -3.5},
                    ],
                    "top": [{"token": "REE"}, {"token": "IVE"}],
                },
                {
                    "name": "copy_block_list_fourth_after_TH",
                    "candidates": [
                        {"label": "correct_REE", "first_token_logprob": -0.1},
                        {"label": "corrupt_IVE", "first_token_logprob": -9.4},
                    ],
                    "top": [{"token": "REE"}],
                },
                {
                    "name": "original_snippet_camera_after_THREE_dot_P",
                    "candidates": [
                        {"label": "correct_ers", "first_token_logprob": -6.9},
                        {"label": "whole_Pers", "first_token_logprob": -0.6},
                    ],
                    "top": [{"token": "Pers"}, {"token": "ers"}],
                },
                {
                    "name": "copy_block_snippet_camera_after_THREE_dot_P",
                    "candidates": [
                        {"label": "correct_ers", "first_token_logprob": -3.0},
                        {"label": "whole_Pers", "first_token_logprob": -3.9},
                    ],
                    "top": [{"token": "ers"}, {"token": "Pers"}],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json",
        {
            "schema": "vmlx-dsv4-direct-vs-thinking-webgl-logit-probe-v1",
            "cases": [
                {
                    "rail": "direct_off",
                    "prefix_name": "after_const_renderer_web",
                    "ranks": {"GL": 1, "Web": 5},
                    "top": [{"token": "GL"}, {"token": "3"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_const_renderer_web",
                    "ranks": {"GL": 1, "Web": 5},
                    "top": [{"token": "GL"}, {"token": "Script"}],
                },
                {
                    "rail": "direct_off",
                    "prefix_name": "after_const_renderer_three_dot",
                    "ranks": {"Web": 4},
                    "top": [{"token": "<｜end▁of▁sentence｜>"}, {"token": "Three"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_const_renderer_three_dot",
                    "ranks": {"Web": 1},
                    "top": [{"token": "Web"}, {"token": "<｜end▁of▁sentence｜>"}],
                },
                {
                    "rail": "direct_off",
                    "prefix_name": "after_camera_p",
                    "ranks": {"ers": 1, "Pers": 2},
                    "top": [{"token": "ers"}, {"token": "Pers"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_camera_p",
                    "ranks": {"Pers": 1, "ers": 2},
                    "top": [{"token": "Pers"}, {"token": "ers"}],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "canonical_return_fences_colon",
                    "exact": True,
                    "normalized_exact": True,
                    "finish_reason": "stop",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "enable_thinking": False,
                    },
                },
                {
                    "name": "return_fences_no_punct",
                    "exact": True,
                    "normalized_exact": True,
                    "finish_reason": "stop",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "enable_thinking": False,
                    },
                },
                {
                    "name": "return_fences_period",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "stop",
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "enable_thinking": False,
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json",
        {
            "schema": "vmlx-dsv4-cache-vs-full-logit-isolation-v1",
            "checks": [
                {
                    "name": "isolated_after_THREE_dot_P",
                    "full_top": [{"token": "ers"}, {"token": "ert"}],
                    "incremental_top": [{"token": "ers"}, {"token": "ert"}],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-logit-trace-v1",
            "decoded": "THREE.PertiveCamera<｜end▁of▁sentence｜>",
            "events": [
                {
                    "generated_so_far_text": "THREE.P",
                    "top": [{"token": "ert"}, {"token": "ers"}],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-warmup-ablation-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-warmup-ablation-v1",
            "runs": [
                {
                    "label": "normal_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-token-tail-source-ab-20260525.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "chat_on_rep1",
                    "route": "chat",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                },
                {
                    "name": "responses_on_rep1",
                    "route": "responses",
                    "exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-rep1-prefill-logits-eval-20260525.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json",
        {
            "status": "dry_run",
            "case_count": 1,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "request_overrides": {
                        "skip_prefix_cache": True,
                        "temperature": 0,
                        "top_p": 1,
                        "repetition_penalty": 1.0,
                    },
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-bundle-defaults-source-20260525.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chat_off_bundle_defaults",
                    "route": "chat",
                    "exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
        {
            "status": "skipped",
            "reason": "insufficient_free_memory",
            "required_available_gb": 120.0,
            "model": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
            "selected_cases": [
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
            ],
            "case_count": 14,
            "telemetry": [
                {
                    "name": "preflight",
                    "system_memory": {
                        "available_gb": 105.89,
                        "total_gb": 128.0,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json",
        {
            "status": "fail",
            "env_overrides": {
                "DSV4_POOL_QUANT": "0",
                "VMLINUX_DSV4_ENABLE_PREFIX_CACHE": "0",
                "PYTHONPATH": "",
                "PYTHONNOUSERSITE": "1",
            },
            "selected_cases": ["chat_off_rep1", "responses_off_rep1"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "content": "const renderer = new THREE.WebWebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "content": "const renderer = new THREE.WebWebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    summary = quality["details"]["exact_code_root_boundary"]
    evidence_joined = "\n".join(quality["evidence"])

    assert quality["status"] == "open"
    assert not any(
        "docs/internal/release-gates/20260520_sisyphus" in item
        for item in quality["details"]["missing_evidence"]
    )
    assert "docs/internal/release-gates/20260520_sisyphus" not in evidence_joined
    assert summary["direct_off_route_wide_failure"] is True
    assert summary["requested_thinking_is_diagnostic_only"] is True
    assert summary["template_mismatch_ruled_out"] is True
    assert summary["hidden_reasoning_not_sufficient_root_cause"] is True
    assert summary["prefix_cache_not_sufficient_root_cause"] is True
    assert summary["forced_sampler_controls_not_sufficient_root_cause"] is True
    assert summary["bundle_defaults_do_not_clear_direct_off"] is True
    assert summary["true_bundled_jangtqk_direct_off_still_fails"] is True
    assert summary["true_bundled_jangtqk_direct_off_recheck"]["env_overrides"][
        "VMLINUX_DSV4_ENABLE_PREFIX_CACHE"
    ] == "0"
    assert summary["true_bundled_jangtqk_direct_off_recheck"][
        "failed_thinking_closed_cases"
    ] == ["chat_off_rep1", "responses_off_rep1"]
    assert summary["batch_generator_intrinsic_failure_not_proven"] is True
    assert summary["stream_warmup_state_ruled_out_for_isolated_prefix"] is True
    assert summary["prompt_wording_changes_identifier_logits"] is True
    assert summary["current_primary_failure"] == "direct_off_exact_code_generation"
    assert summary["source_full_output_preflight"]["artifact_present"] is True
    assert summary["source_full_output_preflight"]["artifact"] == (
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json"
    )
    assert summary["source_full_output_preflight"]["status"] == "skipped"
    assert summary["source_full_output_preflight"]["reason"] == "insufficient_free_memory"
    assert summary["source_full_output_preflight"]["available_gb"] == 105.89
    assert summary["source_full_output_preflight"]["memory_gap_gb"] == 14.11
    assert summary["source_full_output_preflight"]["did_not_launch"] is True
    assert summary["source_full_output_preflight"]["launch_decision"] == "do_not_launch"
    assert summary["source_full_output_preflight"]["required_available_gb"] == 120.0
    assert summary["source_full_output_clearance_missing"] is True
    joined_evidence = "\n".join(quality["evidence"])
    assert (
        "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json"
        in joined_evidence
    )
    assert (
        "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0625.json"
        not in joined_evidence
    )
    assert (
        "current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json"
        in joined_evidence
    )
    assert "true_bundled_jangtqk_direct_off_recheck" in summary[
        "ruled_out_root_causes"
    ]
    assert summary["direct_off_identifier_logit_divergence"] == {
        "webweb_not_explained_by_after_web_rank": True,
        "direct_after_three_dot_web_rank": 4,
        "thinking_after_three_dot_web_rank": 1,
        "direct_after_three_dot_top_token": "<｜end▁of▁sentence｜>",
        "thinking_after_three_dot_top_token": "Web",
        "direct_after_camera_p_top_token": "ers",
        "thinking_after_camera_p_top_token": "Pers",
    }
    assert summary["direct_off_three_dot_identifier_boundary"] is True
    assert summary["prompt_boundary_bisection"] == {
        "canonical_colon_passes": True,
        "no_punct_after_fences_still_passes": True,
        "period_after_fences_breaks_exactness": True,
        "same_effective_rail": True,
        "passing_cases": [
            "canonical_return_fences_colon",
            "return_fences_no_punct",
        ],
        "failed_cases": ["return_fences_period"],
        "effective_rails": ["thinking_open"],
        "all_effective_policy_reasons": ["dsv4_direct_rail_identifier_unsafe"],
    }
    assert summary["direct_off_prompt_boundary_is_wording_sensitive"] is True
    assert summary["prompt_boundary_passes_are_effective_thinking_open"] is True
    assert summary["prompt_boundary_passes_clear_direct_off"] is False
    assert summary["prompt_boundary_requested_off_but_effective_force_on"] is True
    assert "template_mismatch" in summary["ruled_out_root_causes"]
    assert "hidden_reasoning_corruption" in summary["ruled_out_root_causes"]
    assert "prefix_cache" in summary["ruled_out_root_causes"]
    assert "forced_sampling_controls" in summary["ruled_out_root_causes"]


def test_objective_proof_digest_surfaces_current_source_dsv4_bundle_defaults(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-bundle-defaults-source-20260525.json",
        {
            "status": "fail",
            "selected_cases": [
                "chat_off_bundle_defaults",
                "responses_off_bundle_defaults",
            ],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_bundle_defaults",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": [],
                    "content": "const renderer = new THREE.WebJSGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "token_count": 90,
                            "tail_ids": [128804, 128822],
                        },
                    },
                },
                {
                    "name": "responses_off_bundle_defaults",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "missing": ["THREE.WebGLRenderer"],
                    "corrupt_patterns": [],
                    "content": "const renderer = new THREE.WebJSGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                        "token_diagnostics": {
                            "token_count": 90,
                            "tail_ids": [128804, 128822],
                        },
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-bundle-defaults-dryrun-20260525.json",
        {
            "status": "dry_run",
            "selected_cases": [
                "chat_off_bundle_defaults",
                "responses_off_bundle_defaults",
            ],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_bundle_defaults",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
                {
                    "name": "responses_off_bundle_defaults",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    bundle_defaults = quality["details"][
        "current_source_bundle_defaults_exactness_subset"
    ]
    assert quality["status"] == "open"
    assert bundle_defaults["status"] == "fail"
    assert bundle_defaults["dry_run_artifact"].endswith(
        "current-dsv4-route-mode-code-exactness-bundle-defaults-dryrun-20260525.json"
    )
    assert bundle_defaults["dry_run_present"] is True
    assert bundle_defaults["failed_thinking_closed_cases"] == [
        "chat_off_bundle_defaults",
        "responses_off_bundle_defaults",
    ]
    assert bundle_defaults["failed_cases"] == [
        "chat_off_bundle_defaults",
        "responses_off_bundle_defaults",
    ]
    assert bundle_defaults["case_summaries"][0]["missing_identifiers"] == [
        "THREE.WebGLRenderer"
    ]
    assert "WebJSGLRenderer" in bundle_defaults["case_summaries"][0]["content"]
    assert bundle_defaults["case_summaries"][0]["token_tail_ids"] == [
        128804,
        128822,
    ]
    assert bundle_defaults["case_summaries"][0]["token_count"] == 90


def test_objective_proof_digest_surfaces_dsv4_chatmax_budget_stop_rail_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chatmax_1024_budget",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "stop",
                    "missing": ["THREE.PerspectiveCamera"],
                    "corrupt_patterns": ["BBoxGeometry"],
                    "completion_tokens": 445,
                    "content": "const camera = new THREE.PersPerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                        "reasoning_effort": None,
                    },
                },
                {
                    "name": "chatmax_1024_stop_roles",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "stop",
                    "missing": ["THREE.PerspectiveCamera"],
                    "corrupt_patterns": ["BBoxGeometry"],
                    "completion_tokens": 445,
                    "content": "const camera = new THREE.PersPerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
                {
                    "name": "responses_chatmax_1024",
                    "route": "responses",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "completed",
                    "missing": ["THREE.PerspectiveCamera"],
                    "corrupt_patterns": ["BBoxGeometry"],
                    "completion_tokens": 445,
                    "content": "const camera = new THREE.PersPerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
                {
                    "name": "completion_chatmax_512",
                    "route": "completion",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "stop",
                    "missing": ["THREE.PerspectiveCamera"],
                    "corrupt_patterns": ["BBoxGeometry"],
                    "completion_tokens": 445,
                    "content": "const camera = new THREE.PersPerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_completion_chat_rail_identifier_unsafe",
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_chatmax_budget_stop_rail_probe"]

    assert quality["status"] == "open"
    assert probe["status"] == "fail"
    assert probe["all_failed_cases_same_completion_tokens"] is True
    assert probe["larger_budget_ruled_out"] is True
    assert probe["role_stops_ruled_out"] is True
    assert probe["responses_route_also_fails"] is True
    assert probe["completion_route_also_fails"] is True
    assert probe["failed_cases"] == [
        "chatmax_1024_budget",
        "chatmax_1024_stop_roles",
        "responses_chatmax_1024",
        "completion_chatmax_512",
    ]


def test_objective_proof_digest_surfaces_dsv4_prompt_boundary_bisection_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "canonical_return_fences_colon",
                    "exact": True,
                    "normalized_exact": True,
                    "finish_reason": "stop",
                    "completion_tokens": 195,
                    "content": "const scene = new THREE.Scene();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
                {
                    "name": "return_fences_no_punct",
                    "exact": True,
                    "normalized_exact": True,
                    "finish_reason": "stop",
                    "completion_tokens": 195,
                    "content": "const scene = new THREE.Scene();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
                {
                    "name": "return_fences_period",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "stop",
                    "missing": ["THREE.Scene", "THREE.PerspectiveCamera"],
                    "corrupt_patterns": ["PPerspectiveCamera"],
                    "completion_tokens": 195,
                    "content": "const camera = new THREE.PPerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
                {
                    "name": "return_markdown_colon_no_fences",
                    "exact": False,
                    "normalized_exact": False,
                    "finish_reason": "length",
                    "missing": ["THREE.Scene"],
                    "corrupt_patterns": [],
                    "completion_tokens": 512,
                    "content": "",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_prompt_boundary_bisection_probe"]

    assert quality["status"] == "open"
    assert probe["status"] == "fail"
    assert probe["passing_cases"] == [
        "canonical_return_fences_colon",
        "return_fences_no_punct",
    ]
    assert probe["failed_cases"] == [
        "return_fences_period",
        "return_markdown_colon_no_fences",
    ]
    assert probe["period_after_fences_breaks_exactness"] is True
    assert probe["no_punct_after_fences_still_passes"] is True
    assert probe["length_failed_cases"] == ["return_markdown_colon_no_fences"]
    assert probe["same_effective_rail"] is True


def test_objective_proof_digest_surfaces_dsv4_colon_period_logprob_trace(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json",
        {
            "status": "fail",
            "health_before": {"scheduler": {"cache_hit_tokens": 0}},
            "health_after": {"scheduler": {"cache_hit_tokens": 0}},
            "cases": [
                {
                    "name": "colon_pass",
                    "exact": True,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "logprob_entry_count": 195,
                    "content": "const scene = new THREE.Scene();",
                },
                {
                    "name": "period_fail",
                    "exact": False,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "logprob_entry_count": 195,
                    "first_char_diff_index": 26,
                    "missing": ["THREE.Scene"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "content": "const scene = new THREE.Sc();",
                    "diff_token_entry": {"token": " return"},
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json",
        {
            "status": "fail",
            "health_before": {"scheduler": {"cache_hit_tokens": 0}},
            "health_after": {"scheduler": {"cache_hit_tokens": 0}},
            "cases": [
                {
                    "name": "colon_pass",
                    "exact": True,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "logprob_entry_count": 195,
                    "first_visible_diff_index": None,
                    "content": "const scene = new THREE.Scene();",
                    "joined_logprob_text_prefix": "THREE.WebWebGLRenderer</think>const scene = new THREE.Scene();",
                },
                {
                    "name": "period_fail",
                    "exact": False,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "logprob_entry_count": 195,
                    "first_visible_diff_index": 26,
                    "missing": ["THREE.Scene"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "content": "const scene = new THREE.Sc();",
                    "visible_diff_token_index": 124,
                    "visible_diff_token_context": [
                        {
                            "index": 124,
                            "token": "()",
                            "top_logprobs": [{"token": "()"}],
                        }
                    ],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_colon_period_logprob_trace"]

    assert quality["status"] == "open"
    assert probe["visible_status"] == "fail"
    assert probe["raw_status"] == "fail"
    assert probe["same_prompt_tokens"] is True
    assert probe["same_completion_tokens"] is True
    assert probe["cache_hit_tokens_zero"] is True
    assert probe["colon_hidden_thinking_contains_corruption"] is True
    assert probe["period_first_visible_diff_index"] == 26
    assert probe["period_visible_diff_token_index"] == 124
    assert probe["period_corrupt_patterns"] == ["WebWebGLRenderer"]


def test_objective_proof_digest_surfaces_dsv4_scene_token_rank_contrast(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-scene-token-rank-contrast-live-20260524-1324.json",
        {
            "status": "fail",
            "health_before": {"scheduler": {"cache_hit_tokens": 0}},
            "health_after": {"scheduler": {"cache_hit_tokens": 0}},
            "cases": [
                {
                    "name": "colon_pass",
                    "exact": True,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "scene_token_index": 123,
                    "scene_token_abs_index": 529,
                    "scene_token_context": [
                        {
                            "index": 123,
                            "token": ".Sc",
                            "logprob": -0.001,
                            "top_logprobs": [{"token": ".Sc", "logprob": -0.001}],
                        },
                        {
                            "index": 124,
                            "token": "ene",
                            "logprob": -0.1245,
                            "top_logprobs": [
                                {"token": "ene", "logprob": -0.1245},
                                {"token": "();\n", "logprob": -2.7356},
                            ],
                        }
                    ],
                },
                {
                    "name": "period_fail",
                    "exact": False,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "scene_token_index": 123,
                    "scene_token_abs_index": 529,
                    "missing": ["THREE.Scene"],
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "scene_token_context": [
                        {
                            "index": 123,
                            "token": ".Sc",
                            "logprob": -0.001,
                            "top_logprobs": [{"token": ".Sc", "logprob": -0.001}],
                        },
                        {
                            "index": 124,
                            "token": "();\n",
                            "logprob": -0.7578,
                            "top_logprobs": [
                                {"token": "();\n", "logprob": -0.7578},
                                {"token": "ene", "logprob": -1.1028},
                            ],
                        }
                    ],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_scene_token_rank_contrast"]

    assert quality["status"] == "open"
    assert probe["status"] == "fail"
    assert probe["same_prompt_tokens"] is True
    assert probe["same_completion_tokens"] is True
    assert probe["cache_hit_tokens_zero"] is True
    assert probe["scene_token_index"] == 123
    assert probe["rank_contrast_token_index"] == 124
    assert probe["colon_selected_token"] == "ene"
    assert probe["period_selected_token"] == "();\n"
    assert probe["colon_correct_ene_rank"] == 1
    assert probe["colon_wrong_close_rank"] == 2
    assert probe["period_wrong_close_rank"] == 1
    assert probe["period_correct_ene_rank"] == 2


def test_objective_proof_digest_surfaces_dsv4_hidden_reasoning_control(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json",
        {
            "status": "fail",
            "health_before": {"scheduler": {"cache_hit_tokens": 0}},
            "health_after": {"scheduler": {"cache_hit_tokens": 0}},
            "diagnostic_summary": {
                "colon_control_exact": True,
                "period_control_exact": False,
                "system_no_draft_exact": False,
                "system_verify_exact": False,
            },
            "cases": [
                {
                    "name": "colon_known_pass_control",
                    "exact": True,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "hidden_contains_corruption": True,
                    "corrupt_patterns": [],
                    "missing": [],
                },
                {
                    "name": "period_known_fail_control",
                    "exact": False,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "hidden_contains_corruption": True,
                    "corrupt_patterns": ["WebWebGLRenderer"],
                    "missing": ["THREE.Scene"],
                },
                {
                    "name": "period_no_reasoning_code_draft_system",
                    "exact": False,
                    "prompt_tokens": 130,
                    "completion_tokens": 144,
                    "hidden_contains_corruption": False,
                    "corrupt_patterns": ["PPerspectiveCamera"],
                    "missing": ["THREE.PerspectiveCamera"],
                },
                {
                    "name": "period_verify_identifiers_system",
                    "exact": False,
                    "prompt_tokens": 140,
                    "completion_tokens": 401,
                    "hidden_contains_corruption": True,
                    "corrupt_patterns": ["PPerspectiveCamera"],
                    "missing": ["THREE.PerspectiveCamera"],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_hidden_reasoning_control_probe"]

    assert quality["status"] == "open"
    assert probe["status"] == "fail"
    assert probe["colon_control_exact"] is True
    assert probe["period_control_exact"] is False
    assert probe["system_controls_still_fail"] is True
    assert probe["no_draft_hidden_corruption_removed_but_visible_still_failed"] is True
    assert probe["hidden_reasoning_not_sufficient_root_cause"] is True
    assert probe["cache_hit_tokens_zero"] is True
    assert probe["failed_cases"] == [
        "period_known_fail_control",
        "period_no_reasoning_code_draft_system",
        "period_verify_identifiers_system",
    ]


def test_objective_proof_digest_surfaces_dsv4_template_parity(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-template-parity-diagnostic-20260524-1343.json",
        {
            "schema": "vmlx-dsv4-template-parity-diagnostic-v1",
            "all_sidecar_equals_tokenizer_template": True,
            "cases": [
                {
                    "name": "period_exact_code",
                    "enable_thinking": False,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_closed",
                    "boundary_tokens_tail": [
                        "Ġno",
                        "Ġmark",
                        "down",
                        "Ġfences",
                        ".Ċ",
                        "const",
                    ],
                },
                {
                    "name": "colon_exact_code",
                    "enable_thinking": False,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_closed",
                    "boundary_tokens_tail": [
                        "Ġno",
                        "Ġmark",
                        "down",
                        "Ġfences",
                        ":Ċ",
                        "const",
                    ],
                },
                {
                    "name": "period_exact_code",
                    "enable_thinking": True,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_open",
                    "boundary_tokens_tail": [".Ċ", "const"],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_template_parity_diagnostic"]

    assert quality["status"] == "open"
    assert probe["present"] is True
    assert probe["all_sidecar_equals_tokenizer_template"] is True
    assert probe["mismatch_count"] == 0
    assert probe["normal_colon_period_prompt_tokens_equal"] is True
    assert probe["period_boundary_token"] == ".Ċ"
    assert probe["colon_boundary_token"] == ":Ċ"
    assert probe["template_mismatch_ruled_out"] is True


def test_objective_proof_digest_surfaces_dsv4_prefill_execution_variant_logits(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-prefill-execution-variant-logits-20260524.json",
        {
            "schema": "vmlx-dsv4-prefill-execution-variant-logits-v1",
            "target_context": "after generated prefix THREE.P",
            "variants": [
                {
                    "name": "plain_full_prompt_then_prefix",
                    "top": [
                        {"token": "ers", "logprob": -1.38},
                        {"token": "ert", "logprob": -1.46},
                        {"token": "Pers", "logprob": -3.82},
                    ],
                },
                {
                    "name": "stream_full_prompt_then_prefix",
                    "top": [
                        {"token": "ers", "logprob": -1.38},
                        {"token": "ert", "logprob": -1.46},
                        {"token": "Pers", "logprob": -3.82},
                    ],
                },
                {
                    "name": "warmup_stream_clear_after_each_prefix",
                    "top": [
                        {"token": "ers", "logprob": -1.38},
                        {"token": "ert", "logprob": -1.46},
                        {"token": "Pers", "logprob": -3.82},
                    ],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_prefill_execution_variant_logits"]

    assert quality["status"] == "open"
    assert probe["present"] is True
    assert probe["target_context"] == "after generated prefix THREE.P"
    assert probe["variant_count"] == 3
    assert probe["all_variants_same_top_tokens"] is True
    assert probe["all_variants_select_correct_ers"] is True
    assert probe["whole_pers_never_top"] is True
    assert probe["stream_warmup_state_ruled_out_for_isolated_prefix"] is True


def test_objective_proof_digest_surfaces_dsv4_prompt_variant_logit_probe(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-prompt-variant-logit-probe-20260524.json",
        {
            "schema": "vmlx-dsv4-prompt-variant-logit-probe-v1",
            "cases": [
                {
                    "name": "original_list_fourth_after_TH",
                    "prompt_token_count": 47,
                    "prefix": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTH",
                    "candidates": [
                        {
                            "label": "correct_REE",
                            "text": "REE",
                            "first_token_logprob": -0.711,
                        },
                        {
                            "label": "corrupt_IVE",
                            "text": "IVE",
                            "first_token_logprob": -3.506,
                        },
                    ],
                    "top": [
                        {"token": "REE", "logprob": -0.711},
                        {"token": "RE", "logprob": -1.184},
                        {"token": "IVE", "logprob": -3.506},
                    ],
                },
                {
                    "name": "copy_block_list_fourth_after_TH",
                    "prompt_token_count": 55,
                    "prefix": "THREE.Scene\nTHREE.WebGLRenderer\nTHREE.PerspectiveCamera\nTH",
                    "candidates": [
                        {
                            "label": "correct_REE",
                            "text": "REE",
                            "first_token_logprob": -0.141,
                        },
                        {
                            "label": "corrupt_IVE",
                            "text": "IVE",
                            "first_token_logprob": -9.457,
                        },
                    ],
                    "top": [
                        {"token": "REE", "logprob": -0.141},
                        {"token": "RE", "logprob": -2.386},
                    ],
                },
                {
                    "name": "original_snippet_camera_after_THREE_dot_P",
                    "prompt_token_count": 75,
                    "prefix": "const camera = new THREE.P",
                    "candidates": [
                        {
                            "label": "correct_ers",
                            "text": "ers",
                            "first_token_logprob": -6.981,
                        },
                        {
                            "label": "whole_Pers",
                            "text": "Pers",
                            "first_token_logprob": -0.670,
                        },
                    ],
                    "top": [
                        {"token": "Pers", "logprob": -0.670},
                        {"token": "Three", "logprob": -1.192},
                        {"token": "ers", "logprob": -6.981},
                    ],
                },
                {
                    "name": "copy_block_snippet_camera_after_THREE_dot_P",
                    "prompt_token_count": 85,
                    "prefix": "const camera = new THREE.P",
                    "candidates": [
                        {
                            "label": "correct_ers",
                            "text": "ers",
                            "first_token_logprob": -3.039,
                        },
                        {
                            "label": "whole_Pers",
                            "text": "Pers",
                            "first_token_logprob": -3.900,
                        },
                    ],
                    "top": [
                        {"token": "erm", "logprob": -1.876},
                        {"token": "ers", "logprob": -3.039},
                        {"token": "Pers", "logprob": -3.900},
                    ],
                },
                {
                    "name": "original_snippet_box_after_THREE_dot_B",
                    "prompt_token_count": 75,
                    "prefix": "const cube = new THREE.Mesh(new THREE.B",
                    "candidates": [
                        {
                            "label": "correct_ox",
                            "text": "ox",
                            "first_token_logprob": -1.100,
                        },
                        {
                            "label": "whole_Box",
                            "text": "Box",
                            "first_token_logprob": -0.418,
                        },
                    ],
                    "top": [
                        {"token": "Box", "logprob": -0.418},
                        {"token": "ox", "logprob": -1.100},
                    ],
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_prompt_variant_logit_probe"]

    assert quality["status"] == "open"
    assert probe["present"] is True
    assert probe["schema"] == "vmlx-dsv4-prompt-variant-logit-probe-v1"
    assert probe["case_count"] == 5
    assert probe["original_list_correct_REE_rank"] == 1
    assert probe["copy_block_list_correct_REE_rank"] == 1
    assert probe["copy_block_reduces_list_corrupt_IVE_logprob"] is True
    assert probe["original_camera_whole_Pers_rank"] == 1
    assert probe["original_camera_correct_ers_rank"] == 3
    assert probe["copy_block_camera_correct_ers_rank"] == 2
    assert probe["copy_block_camera_whole_Pers_rank"] == 3
    assert probe["copy_block_improves_camera_correct_ers_logprob"] is True
    assert probe["copy_block_reduces_camera_whole_Pers_logprob"] is True
    assert probe["original_box_whole_Box_rank"] == 1
    assert probe["original_box_correct_ox_rank"] == 2
    assert probe["prompt_wording_changes_identifier_logits"] is True


def test_objective_proof_digest_surfaces_dsv4_reasoning_policy_live(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-reasoning-policy-live-20260524-1408.json",
        {
            "schema": "vmlx-dsv4-reasoning-policy-live-v1",
            "status": "pass",
            "cases": [
                {
                    "name": "one_token_chat_explicit_off",
                    "content": "```",
                    "content_is_null": False,
                    "reasoning_chars": 0,
                },
                {
                    "name": "short_chat_explicit_off",
                    "content": "OK",
                    "content_is_null": False,
                    "reasoning_chars": 0,
                },
                {
                    "name": "short_chat_requested_on",
                    "content": None,
                    "content_is_null": True,
                    "reasoning_chars": 36,
                },
            ],
            "health_after": {
                "status": "healthy",
                "model_loaded": True,
                "num_running": 0,
                "num_waiting": 0,
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_reasoning_policy_live"]

    assert quality["status"] == "open"
    assert probe["status"] == "pass"
    assert probe["explicit_off_visible_no_reasoning"] is True
    assert probe["one_token_off_visible_no_reasoning"] is True
    assert probe["requested_on_reasoning_observed"] is True
    assert probe["requested_on_content_is_null"] is True


def test_objective_proof_digest_surfaces_dsv4_batch_generator_logit_divergence(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json",
        {
            "schema": "vmlx-dsv4-cache-vs-full-logit-isolation-v1",
            "checks": [
                {
                    "name": "isolated_after_THREE_dot_P",
                    "full_top": [
                        {"token": "ers", "logprob": -1.001},
                        {"token": "ert", "logprob": -1.676},
                    ],
                    "incremental_top": [
                        {"token": "ers", "logprob": -1.380},
                        {"token": "ert", "logprob": -1.460},
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-logit-trace-v1",
            "decoded": "THREE.PertiveCamera<｜end▁of▁sentence｜>",
            "events": [
                {
                    "generated_so_far_text": "THREE.P",
                    "top": [
                        {"token": "ert", "logprob": -1.228},
                        {"token": "ers", "logprob": -1.730},
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-warmup-ablation-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-warmup-ablation-v1",
            "runs": [
                {
                    "label": "normal_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                },
                {
                    "label": "skip_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    probe = quality["details"]["current_batch_generator_logit_divergence"]

    assert quality["status"] == "open"
    assert probe["direct_artifact_present"] is True
    assert probe["batch_artifact_present"] is True
    assert probe["direct_full_THREE_P_selected"] == "ers"
    assert probe["direct_incremental_THREE_P_selected"] == "ers"
    assert probe["batch_THREE_P_selected"] == "ert"
    assert probe["batch_decoded_contains_corrupt_pertive"] is True
    assert probe["direct_vs_batch_THREE_P_diverges"] is True
    assert probe["warmup_ablation_artifact_present"] is True
    assert probe["warmup_ablation_exact_runs"] == ["normal_warmup", "skip_warmup"]
    assert probe["batch_generator_intrinsic_failure_not_proven"] is True
    assert {
        "gate": "current_batch_generator_logit_divergence",
        "artifact": (
            "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json"
        ),
        "status": "fail",
    } in quality["details"]["failed_quality_gates"]


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


def test_objective_proof_digest_rejects_shallow_default_cache_tool_loop_proof(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop/result.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["checks"] = {
        "tool_sequence_ordered": False,
        "final_done": True,
        "file_written": True,
        "code_file_written_exact": False,
        "cached_tokens_seen": True,
        "dsv4_cache_detail_seen": True,
    }
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    assert row["status"] == "open"
    assert row["details"]["tool_loop_checks"]["tool_sequence_ordered"] is False
    assert row["details"]["tool_loop_checks"]["code_file_written_exact"] is False


def test_objective_proof_digest_surfaces_default_cache_tool_loop_round_diagnostics(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop/result.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["rounds"][1]["output_text"] = "<｜DSML｜tool_c"
    proof["rounds"][1]["response_diagnostics"] = {
        "response_status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_item_statuses": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "incomplete",
                "finish_reason": "length",
                "name": None,
            }
        ],
    }
    proof["checks"]["tool_sequence_ordered"] = False
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    assert row["status"] == "open"
    assert row["details"]["tool_loop_round_outputs"][1] == "<｜DSML｜tool_c"
    assert row["details"]["tool_loop_round_response_diagnostics"][1] == {
        "response_status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_item_statuses": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "incomplete",
                "finish_reason": "length",
                "name": None,
            }
        ],
    }


def test_objective_proof_digest_accepts_default_cache_tool_loop_mechanics_despite_code_diff(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop/result.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["code_tool_probe"] = {
        "expected_content": (
            "const renderer = new THREE.WebGLRenderer();\n"
            "const camera = new THREE.PerspectiveCamera();\n"
        ),
        "actual_content": (
            "const renderer = new THREE.WebRenderer();\n"
            "const camera = new THREE.PerspectiveCamera();\n"
        ),
        "first_difference": {
            "index": 33,
            "expected_char": "G",
            "actual_char": "R",
        },
        "missing_expected_fragments": ["THREE.WebGLRenderer"],
        "corrupt_identifier_patterns": ["THREE.WebRenderer"],
    }
    proof["code_round_request_controls"] = {
        "round": 2,
        "max_output_tokens": 768,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
    }
    proof["checks"]["code_file_written_exact"] = False
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    assert row["status"] == "pass"
    assert row["details"]["code_tool_expected_content"].startswith(
        "const renderer = new THREE.WebGLRenderer();"
    )
    assert row["details"]["code_tool_actual_content"].startswith(
        "const renderer = new THREE.WebRenderer();"
    )
    assert row["details"]["code_tool_missing_identifiers"] == [
        "THREE.WebGLRenderer"
    ]
    assert row["details"]["code_tool_corrupt_identifier_patterns"] == [
        "THREE.WebRenderer"
    ]
    assert row["details"]["code_tool_first_difference"] == {
        "index": 33,
        "expected_char": "G",
        "actual_char": "R",
    }
    assert row["details"]["code_round_request_controls"]["enable_thinking"] is False
    assert row["details"]["code_round_request_controls"]["max_output_tokens"] == 768
    assert row["details"]["failed_required_tool_loop_checks"] == []
    assert row["details"]["tool_loop_checks"]["code_file_written_exact"] is False


def test_objective_proof_digest_surfaces_default_cache_tool_loop_dry_run_controls(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-default-cache-tool-loop-dryrun-controls-20260525.json",
        {
            "status": "dry_run",
            "code_round_request_controls": {
                "round": 2,
                "max_output_tokens": 768,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
                "tools_enabled": True,
                "tool_choice": "auto",
            },
            "code_tool_probe": {
                "expected_content": "const renderer = new THREE.WebGLRenderer();",
                "actual_content": None,
                "exact": False,
                "first_difference": {
                    "index": 0,
                    "expected_char": "c",
                    "actual_char": None,
                },
                "missing_expected_fragments": ["THREE.WebGLRenderer"],
                "corrupt_identifier_patterns": [],
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    diagnostic = row["details"]["default_cache_tool_loop_dry_run_controls"]
    assert diagnostic["artifact_status"] == "dry_run"
    assert diagnostic["artifact_present"] is True
    assert diagnostic["code_round_request_controls"]["max_output_tokens"] == 768
    assert diagnostic["code_round_request_controls"]["enable_thinking"] is False
    assert diagnostic["code_round_request_controls"]["tools_enabled"] is True
    assert diagnostic["code_tool_probe"]["missing_expected_fragments"] == [
        "THREE.WebGLRenderer"
    ]


def test_objective_proof_digest_surfaces_explicit_thinking_tool_loop_diagnostic(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    diagnostic_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop-thinking-on-20260525.json"
    )
    diagnostic_path.write_text(
        json.dumps(
            {
                "status": "review",
                "request_thinking_mode": {
                    "enable_thinking": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                "checks": {
                    "tool_sequence_ordered": False,
                    "final_done": False,
                    "file_written": True,
                    "code_file_written_exact": False,
                    "native_cache": True,
                    "native_prefix": True,
                    "native_paged": True,
                    "native_l2": True,
                    "generic_tq_kv_off": True,
                    "cached_tokens_seen": True,
                    "dsv4_cache_detail_seen": True,
                },
                "rounds": [
                    {"executed_tools": [{"name": "list_directory"}], "output_text": ""},
                    {"executed_tools": [{"name": "list_directory"}], "output_text": ""},
                    {"executed_tools": [{"name": "write_file"}], "output_text": ""},
                    {
                        "executed_tools": [],
                        "output_text": "<｜DSML｜tool_calls>write_file</｜DSML｜tool_calls>",
                    },
                ],
                "tool_loop_cached_tokens": 328,
                "tool_loop_cache_details": ["paged+dsv4"],
                "code_tool_probe": {
                    "expected_content": "const renderer = new THREE.WebGLRenderer();",
                    "actual_content": None,
                },
            }
        ),
        encoding="utf-8",
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    diagnostic = row["details"]["explicit_thinking_tool_loop_diagnostic"]
    assert diagnostic["artifact_status"] == "review"
    assert diagnostic["request_thinking_mode"]["enable_thinking"] is True
    assert diagnostic["executed_tool_names"] == [
        "list_directory",
        "list_directory",
        "write_file",
    ]
    assert diagnostic["final_text"].startswith("<｜DSML｜tool_calls>")
    assert diagnostic["tool_loop_checks"]["tool_sequence_ordered"] is False
    assert diagnostic["tool_loop_cached_tokens"] == 328


def test_objective_proof_digest_surfaces_no_cache_tool_loop_diagnostic(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    diagnostic_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop-nocache-ab-20260525.json"
    )
    diagnostic_path.write_text(
        json.dumps(
            {
                "status": "review",
                "diagnostic_cache_mode": "disabled",
                "checks": {
                    "tool_sequence_ordered": True,
                    "final_done": True,
                    "file_written": True,
                    "code_file_written_exact": False,
                    "native_cache": True,
                    "native_prefix": False,
                    "native_paged": False,
                    "native_l2": False,
                    "generic_tq_kv_off": True,
                    "cached_tokens_seen": False,
                    "dsv4_cache_detail_seen": False,
                },
                "tool_loop_cached_tokens": 0,
                "tool_loop_cache_details": [],
                "code_round_request_controls": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.0,
                    "enable_thinking": False,
                },
                "code_tool_probe": {
                    "expected_content": "const renderer = new THREE.WebGLRenderer();",
                    "actual_content": "const renderer = new THREE.WebRenderer();",
                    "first_difference": {
                        "index": 33,
                        "expected_char": "G",
                        "actual_char": "R",
                    },
                    "missing_expected_fragments": ["THREE.WebGLRenderer"],
                    "corrupt_identifier_patterns": [
                        "THREE.WebRenderer",
                        "renderer.render(scene, camera)",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    diagnostic = row["details"]["no_cache_tool_loop_diagnostic"]
    assert diagnostic["artifact_status"] == "review"
    assert diagnostic["diagnostic_cache_mode"] == "disabled"
    assert diagnostic["tool_loop_checks"]["code_file_written_exact"] is False
    assert diagnostic["tool_loop_cached_tokens"] == 0
    assert diagnostic["code_tool_first_difference"] == {
        "index": 33,
        "expected_char": "G",
        "actual_char": "R",
    }
    assert diagnostic["code_tool_corrupt_identifier_patterns"] == [
        "THREE.WebRenderer"
    ]
    root_cause = row["details"]["default_cache_tool_loop_root_cause"]
    assert root_cause["tool_parser_and_cache_path_proven"] is True
    assert root_cause["default_cache_exact_code_failed"] is False
    assert root_cause["no_cache_exact_code_also_failed"] is True
    assert root_cause["cache_not_sufficient_root_cause"] is True
    assert root_cause["request_controls_rule_out_forced_sampling_fix"] is True


def test_objective_proof_digest_surfaces_prompt_guard_tool_loop_diagnostic(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    diagnostic_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop-after-invoke-prompt-guard-20260525.json"
    )
    diagnostic_path.write_text(
        json.dumps(
            {
                "status": "review",
                "checks": {
                    "tool_sequence_ordered": True,
                    "final_done": True,
                    "file_written": True,
                    "code_file_written_exact": False,
                    "native_cache": True,
                    "native_prefix": True,
                    "native_paged": True,
                    "native_l2": True,
                    "generic_tq_kv_off": True,
                    "cached_tokens_seen": True,
                    "dsv4_cache_detail_seen": True,
                },
                "tool_loop_cached_tokens": 528,
                "tool_loop_cache_details": ["paged+dsv4"],
                "code_tool_probe": {
                    "expected_content": "const camera = new THREE.PerspectiveCamera();",
                    "actual_content": (
                        "const camera = new THREE.PPerspectiveCamera();\n"
                        "const cube = new THREE.Mesh(new THREE.BBoxGeometry());"
                    ),
                    "first_difference": {
                        "index": 25,
                        "expected_char": "e",
                        "actual_char": "P",
                    },
                    "missing_expected_fragments": ["THREE.PerspectiveCamera"],
                    "corrupt_identifier_patterns": [],
                },
            }
        ),
        encoding="utf-8",
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    diagnostic = row["details"]["prompt_guard_tool_loop_diagnostic"]
    assert diagnostic["artifact_status"] == "review"
    assert diagnostic["tool_loop_checks"]["tool_sequence_ordered"] is True
    assert diagnostic["tool_loop_checks"]["code_file_written_exact"] is False
    assert diagnostic["tool_loop_cached_tokens"] == 528
    assert diagnostic["code_tool_missing_identifiers"] == [
        "THREE.PerspectiveCamera"
    ]
    assert diagnostic["code_tool_corrupt_identifier_patterns"] == [
        "THREE.PPerspectiveCamera",
        "THREE.BBoxGeometry",
    ]


def test_objective_proof_digest_surfaces_copy_block_tool_loop_diagnostic(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    diagnostic_path = (
        tmp_path
        / "build/current-dsv4-default-cache-tool-loop-copy-block-prompt-20260525.json"
    )
    diagnostic_path.write_text(
        json.dumps(
            {
                "status": "review",
                "diagnostic_code_prompt_variant": "copy_block",
                "checks": {
                    "tool_sequence_ordered": True,
                    "final_done": True,
                    "file_written": True,
                    "code_file_written_exact": False,
                    "native_cache": True,
                    "native_prefix": True,
                    "native_paged": True,
                    "native_l2": True,
                    "generic_tq_kv_off": True,
                    "cached_tokens_seen": True,
                    "dsv4_cache_detail_seen": True,
                },
                "tool_loop_cached_tokens": 625,
                "tool_loop_cache_details": ["paged+dsv4"],
                "code_tool_probe": {
                    "expected_content": "const renderer = new THREE.WebGLRenderer();",
                    "actual_content": (
                        "const renderer = new THREE.Web3GLRenderer();\n"
                        "const cube = new THREE.Mesh(new THREE.BBoxGeometry());"
                    ),
                    "first_difference": {
                        "index": 33,
                        "expected_char": "G",
                        "actual_char": "3",
                    },
                    "missing_expected_fragments": ["THREE.WebGLRenderer"],
                    "corrupt_identifier_patterns": ["THREE.BBoxGeometry"],
                },
            }
        ),
        encoding="utf-8",
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["DSV4 default-cache multi-tool agent loop is proven"]
    diagnostic = row["details"]["copy_block_tool_loop_diagnostic"]
    assert diagnostic["artifact_status"] == "review"
    assert diagnostic["diagnostic_code_prompt_variant"] == "copy_block"
    assert diagnostic["tool_loop_checks"]["code_file_written_exact"] is False
    assert diagnostic["tool_loop_cached_tokens"] == 625
    assert diagnostic["code_tool_first_difference"] == {
        "index": 33,
        "expected_char": "G",
        "actual_char": "3",
    }
    assert diagnostic["code_tool_missing_identifiers"] == ["THREE.WebGLRenderer"]
    assert diagnostic["code_tool_corrupt_identifier_patterns"] == [
        "THREE.BBoxGeometry"
    ]


def test_objective_proof_digest_surfaces_thinking_copy_block_tool_loop_attempt(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-dsv4-default-cache-tool-loop-thinking-on-copy-block-768-20260525.json",
        {
            "status": "skipped",
            "reason": "insufficient_free_memory",
            "required_available_gb": 120.0,
            "telemetry": [
                {
                    "name": "preflight",
                    "system_memory": {"available_gb": 106.96},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-default-cache-tool-loop-thinking-on-copy-block-768-dryrun-20260525.json",
        {
            "status": "dry_run",
            "diagnostic_code_prompt_variant": "copy_block",
            "diagnostic_cache_mode": "default_native_prefix_paged_l2",
            "request_max_output_tokens": 768,
            "request_thinking_mode": {
                "enable_thinking": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "code_round_request_controls": {
                "max_output_tokens": 768,
                "enable_thinking": True,
                "chat_template_kwargs": {"enable_thinking": True},
                "tool_choice": "auto",
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    diagnostic = rows["DSV4 default-cache multi-tool agent loop is proven"][
        "details"
    ]["thinking_copy_block_tool_loop_diagnostic"]

    assert diagnostic["artifact_status"] == "skipped"
    assert diagnostic["artifact_reason"] == "insufficient_free_memory"
    assert diagnostic["available_gb"] == 106.96
    assert diagnostic["dry_run_status"] == "dry_run"
    assert diagnostic["dry_run_code_prompt_variant"] == "copy_block"
    assert diagnostic["dry_run_request_max_output_tokens"] == 768
    assert diagnostic["dry_run_request_thinking_mode"] == {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_objective_proof_digest_keeps_gemma4_visible_content_failure_open_without_app_visible_artifact(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json",
        {
            "status": "fail",
            "reason": "non-ok stage(s): 0:response_contract_failed",
            "row": "gemma4_26b_jang4m",
            "request_route": "responses",
            "response_contract": {"expect_visible_content": True},
            "results": [{"status": "response_contract_failed"}],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows[
        "Gemma4 26B CRACK Responses visible-content and language quality is release-cleared"
    ]
    assert row["status"] == "open"
    assert row["details"]["artifact_status"] == "fail"
    assert row["details"]["reason"] == "non-ok stage(s): 0:response_contract_failed"
    assert row["details"]["expect_visible_content"] is True


def test_objective_proof_digest_records_gemma4_visible_root_cause_matrix(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json",
        {
            "status": "pass",
            "row": "gemma4_26b_jang4m",
            "request_route": "responses",
            "max_tokens": 512,
            "response_contract": {"expect_visible_content": True},
            "results": [
                {
                    "status": "ok",
                    "response_rails": {
                        "visible_chars": 517,
                        "reasoning_chars": 1056,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-contract-20260524.json",
        {
            "status": "fail",
            "reason": "non-ok stage(s): 0:response_contract_failed",
            "row": "gemma4_26b_jang4m",
            "request_route": "responses",
            "max_thinking_tokens": 16,
            "response_contract": {"expect_visible_content": True},
            "results": [
                {
                    "status": "response_contract_failed",
                    "response_rails": {
                        "visible_chars": 0,
                        "reasoning_chars": 511,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-nocache-20260524.json",
        {
            "status": "fail",
            "reason": "non-ok stage(s): 0:response_contract_failed",
            "request_route": "responses",
            "cache_proof_policy": {"skip_prefix_cache": True},
            "results": [
                {
                    "status": "response_contract_failed",
                    "response_rails": {
                        "visible_chars": 0,
                        "reasoning_chars": 483,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-512-nocache-20260524.json",
        {
            "status": "pass",
            "request_route": "responses",
            "max_tokens": 512,
            "results": [
                {
                    "status": "ok",
                    "response_rails": {
                        "visible_chars": 739,
                        "reasoning_chars": 958,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingoff-visible-nocache-20260524.json",
        {
            "status": "pass",
            "request_route": "responses",
            "results": [
                {
                    "status": "ok",
                    "response_rails": {
                        "visible_chars": 1177,
                        "reasoning_chars": 0,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingbudget16-visible-nocache-20260524.json",
        {
            "status": "fail",
            "request_route": "chat",
            "results": [
                {
                    "status": "response_contract_failed",
                    "response_rails": {
                        "visible_chars": 0,
                        "reasoning_chars": 483,
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-local-generation-metadata-audit-20260524-gemma4-visible-budget.json",
        {
            "status": "pass",
            "review_notes": {
                "Gemma-4-26B-A4B-it-JANG_4M-CRACK": [
                    "thinking_budget_override_forwarded_but_template_does_not_enforce"
                ]
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows[
        "Gemma4 26B CRACK Responses visible-content and language quality is release-cleared"
    ]
    assert row["status"] == "pass"
    details = row["details"]
    assert details["root_cause_summary"] == (
        "Gemma4 thinking_budget is forwarded but the local template does not "
        "consume it; thinking-on can exhaust a low max_tokens cap before "
        "visible content."
    )
    assert details["cache_bypass_reproduces_missing_visible"] is True
    assert details["higher_output_cap_restores_visible"] is True
    assert details["thinking_off_restores_visible"] is True
    assert details["chat_route_reproduces_missing_visible"] is True
    assert details["metadata_budget_unsupported"] is True
    assert details["primary_visible_ok"] is True
    assert details["diagnostic_artifacts"]["primary_app_visible"]["visible_chars"] == 517
    assert (
        details["diagnostic_artifacts"]["unsupported_budget_contract"]["status"]
        == "fail"
    )
    assert details["diagnostic_artifacts"]["responses_128_nocache"]["status"] == "fail"
    assert details["diagnostic_artifacts"]["responses_512_nocache"]["visible_chars"] == 739


def test_objective_proof_digest_keeps_gemma4_mixed_swa_speed_floor_open(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    for artifact, speeds, generation_tps in (
        (
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
            (59.325, 94.315),
            (104.0, 104.0),
        ),
        (
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
            (41.425, 94.079),
            (104.0, 104.0),
        ),
        (
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
            (58.116,),
            (71.453,),
        ),
        (
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
            (77.074, 77.039),
            (104.0, 104.0),
        ),
        (
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
            (79.602, 100.787),
            (104.0, 104.0),
        ),
    ):
        results = []
        cumulative_tokens = 0
        cumulative_time = 0.0
        for idx, (speed, internal_tps) in enumerate(zip(speeds, generation_tps)):
            tokens = 512
            cumulative_tokens += tokens
            cumulative_time += tokens / internal_tps
            cached = 1674 if idx else 0
            results.append(
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": round(tokens / speed, 3),
                        "decode_tok_s_wall": speed,
                        "cached_tokens": cached,
                        "cache_detail": "paged+mixed_swa" if cached else None,
                    },
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": cumulative_tokens,
                                        "generation_time": cumulative_time,
                                        "generation_tps": internal_tps,
                                    }
                                },
                                "native_cache": {
                                    "schema": "mixed_swa_kv_v1",
                                    "generic_turboquant_kv": {
                                        "enabled": False,
                                        "reason": "not_active",
                                    },
                                },
                            }
                        }
                    },
                }
            )
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": results,
            },
        )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]
    assert row["status"] == "open"
    assert row["details"]["speed_floor_tok_s"] == 80.0
    assert row["details"]["min_decode_tok_s"] == 41.425
    assert row["details"]["all_required_artifacts_present"] is True
    assert row["details"]["all_required_artifacts_pass"] is True
    assert row["details"]["all_required_artifacts_mixed_swa"] is True
    assert all(
        artifact["native_cache_generic_tq_enabled"] is False
        for artifact in row["details"]["artifacts"]
    )
    assert row["details"]["artifacts"][0]["artifact"].endswith(
        "cachehit-256-source-skip-redundant-store-notrace-20260525.json"
    )
    assert row["details"]["artifacts"][0]["cache_details"] == ["paged+mixed_swa"]
    assert row["details"]["artifacts"][1]["artifact"].endswith(
        "cachehit-256-installed-app-skip-redundant-store-20260525.json"
    )
    assert row["details"]["artifacts"][1]["min_decode_tok_s"] == 41.425
    assert row["details"]["artifacts"][1]["wall_vs_generation_time"] == [
        {
            "completion_tokens": 512,
            "wall_seconds": 12.36,
            "internal_generation_seconds": 4.923,
            "overhead_seconds": 7.437,
            "wall_decode_tok_s": 41.425,
            "internal_generation_tok_s": 104.0,
        },
        {
            "completion_tokens": 512,
            "wall_seconds": 5.442,
            "internal_generation_seconds": 4.923,
            "overhead_seconds": 0.519,
            "wall_decode_tok_s": 94.079,
            "internal_generation_tok_s": 104.0,
        },
    ]
    assert row["details"]["artifacts"][2]["artifact"].endswith(
        "speed-floor-installed-app-nocache-20260525.json"
    )
    assert row["details"]["artifacts"][2]["min_decode_tok_s"] == 58.116
    assert row["details"]["artifacts"][3]["artifact"].endswith(
        "speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json"
    )
    assert row["details"]["artifacts"][3]["min_decode_tok_s"] == 77.039
    assert row["details"]["artifacts"][3]["max_overhead_seconds"] == 1.723
    assert row["details"]["artifacts"][4]["artifact"].endswith(
        "speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json"
    )
    assert row["details"]["artifacts"][4]["min_decode_tok_s"] == 79.602
    assert row["details"]["all_internal_generation_speeds_clear_floor"] is False
    assert row["details"]["all_decode_speeds_clear_floor"] is False
    assert row["details"]["all_required_artifacts_have_structured_speed"] is True
    assert row["details"]["max_wall_generation_overhead_seconds"] == 7.437
    assert row["details"]["failed_speed_floor_artifacts"] == [
        {
            "artifact": "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
            "min_decode_tok_s": 59.325,
            "min_internal_generation_tok_s": 104.0,
            "failed_metrics": ["wall_decode"],
        },
        {
            "artifact": "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
            "min_decode_tok_s": 41.425,
            "min_internal_generation_tok_s": 104.0,
            "failed_metrics": ["wall_decode"],
        },
        {
            "artifact": "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
            "min_decode_tok_s": 58.116,
            "min_internal_generation_tok_s": 71.453,
            "failed_metrics": ["wall_decode", "internal_generation"],
        },
        {
            "artifact": "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
            "min_decode_tok_s": 77.039,
            "min_internal_generation_tok_s": 104.0,
            "failed_metrics": ["wall_decode"],
        },
        {
            "artifact": "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
            "min_decode_tok_s": 79.602,
            "min_internal_generation_tok_s": 104.0,
            "failed_metrics": ["wall_decode"],
        },
    ]


def test_objective_proof_digest_surfaces_gemma4_sustained_cachehit_diagnostic(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    diagnostic = (
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-"
        "speed-floor-installed-app-triple-cachehit-trace-20260525.json"
    )
    _write_json(
        tmp_path,
        diagnostic,
        {
            "status": "pass",
            "row": "gemma4_26b_jang4m",
            "request_route": "chat",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "health_ready": {
                "native_cache": {
                    "schema": "mixed_swa_kv_v1",
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "not_active",
                    },
                }
            },
            "results": [
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": 6.651,
                        "decode_tok_s_wall": 76.981,
                        "cached_tokens": 0,
                        "cache_detail": None,
                    },
                    "usage_summary": {
                        "prompt_tokens": 1161,
                        "completion_tokens": 512,
                        "cached_tokens": 0,
                    },
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": 512,
                                        "generation_time": 5.12,
                                        "generation_tps": 100.0,
                                        "prompt_time": 0.64,
                                    }
                                }
                            }
                        }
                    },
                },
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": 5.273,
                        "decode_tok_s_wall": 97.098,
                        "cached_tokens": 1160,
                        "cache_detail": "paged+mixed_swa",
                    },
                    "usage_summary": {
                        "prompt_tokens": 1161,
                        "completion_tokens": 512,
                        "cached_tokens": 1160,
                    },
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": 1024,
                                        "generation_time": 10.24,
                                        "generation_tps": 100.0,
                                        "prompt_time": 0.70,
                                    }
                                }
                            }
                        }
                    },
                },
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": 5.289,
                        "decode_tok_s_wall": 96.805,
                        "cached_tokens": 1160,
                        "cache_detail": "paged+mixed_swa",
                    },
                    "usage_summary": {
                        "prompt_tokens": 1161,
                        "completion_tokens": 512,
                        "cached_tokens": 1160,
                    },
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": 1536,
                                        "generation_time": 15.36,
                                        "generation_tps": 100.0,
                                        "prompt_time": 0.76,
                                    }
                                }
                            }
                        }
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]
    cachehit = row["details"]["sustained_cachehit_diagnostic"]
    assert cachehit["artifact_present"] is True
    assert cachehit["artifact_status"] == "pass"
    assert cachehit["native_cache_schema"] == "mixed_swa_kv_v1"
    assert cachehit["native_cache_generic_tq_enabled"] is False
    assert cachehit["cold_decode_tok_s_wall"] == 76.981
    assert cachehit["cachehit_decode_tok_s_wall"] == [97.098, 96.805]
    assert cachehit["min_cachehit_decode_tok_s_wall"] == 96.805
    assert cachehit["per_request_internal_generation_tps"] == [100.0, 100.0, 100.0]
    assert cachehit["wall_vs_generation_time"] == [
        {
            "completion_tokens": 512,
            "wall_seconds": 6.651,
            "internal_generation_seconds": 5.12,
            "scheduler_prompt_seconds": 0.64,
            "overhead_seconds": 1.531,
            "residual_overhead_seconds": 0.891,
            "wall_decode_tok_s": 76.981,
            "internal_generation_tok_s": 100.0,
        },
        {
            "completion_tokens": 512,
            "wall_seconds": 5.273,
            "internal_generation_seconds": 5.12,
            "scheduler_prompt_seconds": 0.06,
            "overhead_seconds": 0.153,
            "residual_overhead_seconds": 0.093,
            "wall_decode_tok_s": 97.098,
            "internal_generation_tok_s": 100.0,
        },
        {
            "completion_tokens": 512,
            "wall_seconds": 5.289,
            "internal_generation_seconds": 5.12,
            "scheduler_prompt_seconds": 0.06,
            "overhead_seconds": 0.169,
            "residual_overhead_seconds": 0.109,
            "wall_decode_tok_s": 96.805,
            "internal_generation_tok_s": 100.0,
        },
    ]
    assert cachehit["max_overhead_seconds"] == 1.531
    assert cachehit["cachehit_turns_clear_floor"] is True
    assert cachehit["cold_turn_clears_floor"] is False
    assert row["status"] == "open"


def test_objective_proof_digest_rejects_gemma4_speed_clearance_when_current_installed_cold_diagnostic_is_slow(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    speed_artifacts = [
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
    ]
    for artifact in speed_artifacts:
        tokens = 512
        internal_tps = 100.0
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": [
                    {
                        "status": "ok",
                        "speed": {
                            "wall_seconds": round(tokens / 90.0, 3),
                            "decode_tok_s_wall": 90.0,
                            "cached_tokens": 1160,
                            "cache_detail": "paged+mixed_swa",
                        },
                        "after": {
                            "health": {
                                "body": {
                                    "scheduler": {
                                        "batch_generator": {
                                            "generation_tokens": tokens,
                                            "generation_time": tokens / internal_tps,
                                            "generation_tps": internal_tps,
                                        }
                                    },
                                    "native_cache": {
                                        "schema": "mixed_swa_kv_v1",
                                        "generic_turboquant_kv": {
                                            "enabled": False,
                                            "reason": "not_active",
                                        },
                                    },
                                }
                            }
                        },
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-cachehit-trace-20260525.json",
        {
            "status": "pass",
            "row": "gemma4_26b_jang4m",
            "request_route": "chat",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "health_ready": {
                "native_cache": {
                    "schema": "mixed_swa_kv_v1",
                    "generic_turboquant_kv": {"enabled": False},
                }
            },
            "results": [
                {
                    "status": "ok",
                    "speed": {
                        "decode_tok_s_wall": 76.981,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"cached_tokens": 0},
                },
                {
                    "status": "ok",
                    "speed": {
                        "decode_tok_s_wall": 97.098,
                        "cached_tokens": 1160,
                        "cache_detail": "paged+mixed_swa",
                    },
                    "usage_summary": {"cached_tokens": 1160},
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]

    assert row["details"]["all_decode_speeds_clear_floor"] is True
    assert row["details"]["all_internal_generation_speeds_clear_floor"] is True
    assert row["details"]["sustained_cachehit_diagnostic"]["cold_turn_clears_floor"] is False
    assert row["details"]["sustained_cachehit_blocks_clearance"] is True
    assert row["status"] == "open"


def test_objective_proof_digest_surfaces_gemma4_short_nocache_repeat_boundary(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    speed_artifacts = [
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
    ]
    for artifact in speed_artifacts:
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": [
                    {
                        "status": "ok",
                        "speed": {
                            "decode_tok_s_wall": 91.0,
                            "cached_tokens": 1160,
                            "cache_detail": "paged+mixed_swa",
                        },
                        "after": {
                            "health": {
                                "body": {
                                    "scheduler": {
                                        "batch_generator": {
                                            "generation_tokens": 512,
                                            "generation_time": 5.12,
                                            "generation_tps": 100.0,
                                        }
                                    },
                                    "native_cache": {
                                        "schema": "mixed_swa_kv_v1",
                                        "generic_turboquant_kv": {"enabled": False},
                                    },
                                }
                            }
                        },
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-trace-20260525.json",
        {
            "status": "pass",
            "row": "gemma4_26b_jang4m",
            "request_route": "chat",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "results": [
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": 3.289,
                        "decode_tok_s_wall": 77.835,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": 256,
                                        "generation_time": 2.56,
                                        "generation_tps": 100.0,
                                        "prompt_time": 0.64,
                                    }
                                }
                            }
                        }
                    },
                },
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": 3.285,
                        "decode_tok_s_wall": 77.93,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": 512,
                                        "generation_time": 5.12,
                                        "generation_tps": 100.0,
                                        "prompt_time": 1.28,
                                    }
                                }
                            }
                        }
                    },
                },
                {
                    "status": "ok",
                    "speed": {
                        "wall_seconds": 3.294,
                        "decode_tok_s_wall": 77.717,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                    "after": {
                        "health": {
                            "body": {
                                "scheduler": {
                                    "batch_generator": {
                                        "generation_tokens": 768,
                                        "generation_time": 7.68,
                                        "generation_tps": 100.0,
                                        "prompt_time": 1.92,
                                    }
                                }
                            }
                        }
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]
    diagnostic = row["details"]["short_nocache_repeat_diagnostic"]

    assert diagnostic["artifact_present"] is True
    assert diagnostic["artifact_status"] == "pass"
    assert diagnostic["decode_tok_s_wall"] == [77.835, 77.93, 77.717]
    assert diagnostic["min_decode_tok_s_wall"] == 77.717
    assert diagnostic["per_request_internal_generation_tps"] == [100.0, 100.0, 100.0]
    assert diagnostic["min_internal_generation_tok_s"] == 100.0
    assert diagnostic["wall_vs_generation_time"] == [
        {
            "completion_tokens": 256,
            "wall_seconds": 3.289,
            "internal_generation_seconds": 2.56,
            "scheduler_prompt_seconds": 0.64,
            "overhead_seconds": 0.729,
            "residual_overhead_seconds": 0.089,
            "wall_decode_tok_s": 77.835,
            "internal_generation_tok_s": 100.0,
        },
        {
            "completion_tokens": 256,
            "wall_seconds": 3.285,
            "internal_generation_seconds": 2.56,
            "scheduler_prompt_seconds": 0.64,
            "overhead_seconds": 0.725,
            "residual_overhead_seconds": 0.085,
            "wall_decode_tok_s": 77.93,
            "internal_generation_tok_s": 100.0,
        },
        {
            "completion_tokens": 256,
            "wall_seconds": 3.294,
            "internal_generation_seconds": 2.56,
            "scheduler_prompt_seconds": 0.64,
            "overhead_seconds": 0.734,
            "residual_overhead_seconds": 0.094,
            "wall_decode_tok_s": 77.717,
            "internal_generation_tok_s": 100.0,
        },
    ]
    assert diagnostic["max_overhead_seconds"] == 0.734
    assert diagnostic["turns_clear_floor"] is False
    assert row["details"]["short_nocache_repeat_blocks_clearance"] is True
    assert row["status"] == "open"


def test_objective_proof_digest_surfaces_gemma4_short_nocache_scheduler_trace(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    speed_artifacts = [
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
    ]
    for artifact in speed_artifacts:
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": [
                    {
                        "status": "ok",
                        "speed": {
                            "decode_tok_s_wall": 91.0,
                            "cached_tokens": 1160,
                            "cache_detail": "paged+mixed_swa",
                        },
                        "after": {
                            "health": {
                                "body": {
                                    "scheduler": {
                                        "batch_generator": {
                                            "generation_tokens": 512,
                                            "generation_time": 5.12,
                                            "generation_tps": 100.0,
                                        }
                                    },
                                    "native_cache": {
                                        "schema": "mixed_swa_kv_v1",
                                        "generic_turboquant_kv": {"enabled": False},
                                    },
                                }
                            }
                        },
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-trace-20260525.json",
        {
            "status": "pass",
            "results": [
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 77.9, "cached_tokens": 0},
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-scheduler-trace-20260525.json",
        {
            "status": "pass",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "results": [
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 78.025, "cached_tokens": 0},
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                },
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 77.741, "cached_tokens": 0},
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                },
            ],
            "log_tail": [
                "INFO:vmlx_engine.mllm_batch_generator:VMLINUX_MLLM_PREFILL_TRACE request_id=chatcmpl-a prompt_tokens=1161 cached_tokens=0 total_ms=639.577 forward_ms=619.64",
                "INFO:vmlx_engine.mllm_scheduler:VMLINUX_MLLM_SCHEDULER_TRACE request_id=chatcmpl-a steps=256 output_tokens=256 batch_next_ms=3211.007 process_cleanup_ms=0.068 cleanup_finished_ms=2.096",
                "INFO:vmlx_engine.mllm_batch_generator:VMLINUX_DECODE_TRACE_NEXT mllm steps=64 last_total_ms=10.04 last_step_ms=2.03 last_async_ms=8.01 last_materialize_ms=0.01 prompt_processing=False batch=1",
                "INFO:vmlx_engine.mllm_batch_generator:VMLINUX_DECODE_TRACE_NEXT mllm steps=128 last_total_ms=10.01 last_step_ms=2.01 last_async_ms=7.99 last_materialize_ms=0.00 prompt_processing=False batch=1",
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-source-triple-nocache-256-sync-eval-ab-20260525.json",
        {
            "status": "pass",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "results": [
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 68.394, "cached_tokens": 0},
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                },
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 68.688, "cached_tokens": 0},
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                },
            ],
            "log_tail": [
                "INFO:vmlx_engine.mllm_scheduler:VMLINUX_MLLM_SCHEDULER_TRACE request_id=chatcmpl-a steps=256 output_tokens=256 batch_next_ms=3682.977 process_cleanup_ms=0.066 cleanup_finished_ms=1.386",
                "INFO:vmlx_engine.mllm_batch_generator:VMLINUX_DECODE_TRACE_NEXT mllm steps=64 last_total_ms=11.90 last_step_ms=1.31 last_async_ms=10.59 last_materialize_ms=0.00 prompt_processing=False batch=1",
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json",
        {
            "status": "pass",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "request_route": "chat",
            "results": [
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 77.085, "cached_tokens": 0},
                    "stream_speed": {
                        "ttft_seconds": 0.733153,
                        "post_first_token_seconds": 2.58798,
                        "decode_tok_s_stream": 98.532,
                        "completion_tokens": 256,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                },
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 79.012, "cached_tokens": 0},
                    "stream_speed": {
                        "ttft_seconds": 0.654353,
                        "post_first_token_seconds": 2.585948,
                        "decode_tok_s_stream": 98.61,
                        "completion_tokens": 256,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 256, "cached_tokens": 0},
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]
    trace = row["details"]["short_nocache_scheduler_trace"]

    assert trace["artifact_present"] is True
    assert trace["artifact_status"] == "pass"
    assert trace["decode_tok_s_wall"] == [78.025, 77.741]
    assert trace["prefill_total_ms"] == [639.577]
    assert trace["scheduler_batch_next_ms"] == [3211.007]
    assert trace["decode_next_last_total_ms"] == [10.04, 10.01]
    assert trace["decode_next_last_async_ms"] == [8.01, 7.99]
    assert trace["max_decode_next_last_async_ms"] == 8.01
    assert trace["async_decode_wait_dominates"] is True

    sync_eval = row["details"]["short_nocache_sync_eval_ab"]
    assert sync_eval["artifact_present"] is True
    assert sync_eval["artifact_status"] == "pass"
    assert sync_eval["decode_tok_s_wall"] == [68.394, 68.688]
    assert sync_eval["min_decode_tok_s_wall"] == 68.394
    assert sync_eval["turns_clear_floor"] is False
    assert sync_eval["scheduler_batch_next_ms"] == [3682.977]
    assert sync_eval["decode_next_last_async_ms"] == [10.59]

    streaming = row["details"]["short_nocache_streaming_diagnostic"]
    assert streaming["artifact_present"] is True
    assert streaming["artifact_status"] == "pass"
    assert streaming["decode_tok_s_wall"] == [77.085, 79.012]
    assert streaming["decode_tok_s_stream"] == [98.532, 98.61]
    assert streaming["min_decode_tok_s_stream"] == 98.532
    assert streaming["stream_turns_clear_floor"] is True
    assert streaming["wall_turns_clear_floor"] is False
    assert row["status"] == "pass"


def test_objective_proof_digest_rejects_gemma4_streaming_speed_clearance_from_short_burst(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    speed_artifacts = [
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
    ]
    for artifact in speed_artifacts:
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": [
                    {
                        "status": "ok",
                        "speed": {
                            "decode_tok_s_wall": 70.0,
                            "cached_tokens": 0,
                        },
                        "after": {
                            "health": {
                                "body": {
                                    "scheduler": {
                                        "batch_generator": {
                                            "generation_tokens": 256,
                                            "generation_time": 2.56,
                                            "generation_tps": 100.0,
                                        }
                                    },
                                    "native_cache": {
                                        "schema": "mixed_swa_kv_v1",
                                        "generic_turboquant_kv": {"enabled": False},
                                    },
                                }
                            }
                        },
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json",
        {
            "status": "pass",
            "python": "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12",
            "request_route": "chat",
            "results": [
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 70.0, "cached_tokens": 0},
                    "stream_speed": {
                        "decode_tok_s_stream": 120.0,
                        "completion_tokens": 8,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 8, "cached_tokens": 0},
                },
                {
                    "status": "ok",
                    "speed": {"decode_tok_s_wall": 70.0, "cached_tokens": 0},
                    "stream_speed": {
                        "decode_tok_s_stream": 120.0,
                        "completion_tokens": 8,
                        "cached_tokens": 0,
                    },
                    "usage_summary": {"completion_tokens": 8, "cached_tokens": 0},
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]
    streaming = row["details"]["short_nocache_streaming_diagnostic"]

    assert streaming["stream_turns_clear_floor"] is False
    assert streaming["min_completion_tokens"] == 8
    assert row["status"] == "open"


def test_objective_proof_digest_rejects_gemma4_speed_clearance_from_log_tail_only(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    speed_artifacts = [
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
    ]
    for artifact in speed_artifacts:
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": [
                    {
                        "status": "ok",
                        "after": {
                            "health": {
                                "body": {
                                    "native_cache": {"schema": "mixed_swa_kv_v1"}
                                }
                            }
                        },
                    }
                ],
                "log_tail": [
                    "INFO:vmlx_engine.server:Chat completion: 128 tokens in 1.28s (100.0 tok/s)"
                ],
            },
        )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]

    assert row["status"] == "open"
    assert row["details"]["all_decode_speeds_clear_floor"] is True
    assert row["details"]["all_required_artifacts_have_structured_speed"] is False


def test_objective_proof_digest_rejects_gemma4_speed_clearance_when_internal_generation_is_slow(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    speed_artifacts = [
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
    ]
    for index, artifact in enumerate(speed_artifacts):
        tokens = 512
        internal_tps = 70.0 if index == 2 else 100.0
        _write_json(
            tmp_path,
            artifact,
            {
                "status": "pass",
                "row": "gemma4_26b_jang4m",
                "request_route": "chat",
                "results": [
                    {
                        "status": "ok",
                        "speed": {
                            "wall_seconds": round(tokens / 90.0, 3),
                            "decode_tok_s_wall": 90.0,
                            "cached_tokens": 0,
                            "cache_detail": "paged+mixed_swa",
                        },
                        "after": {
                            "health": {
                                "body": {
                                    "scheduler": {
                                        "batch_generator": {
                                            "generation_tokens": tokens,
                                            "generation_time": tokens / internal_tps,
                                            "generation_tps": internal_tps,
                                        }
                                    },
                                    "native_cache": {
                                        "schema": "mixed_swa_kv_v1",
                                        "generic_turboquant_kv": {
                                            "enabled": False,
                                            "reason": "not_active",
                                        },
                                    },
                                }
                            }
                        },
                    }
                ],
            },
        )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows[
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared"
    ]

    assert row["status"] == "open"
    assert row["details"]["all_decode_speeds_clear_floor"] is True
    assert row["details"]["all_internal_generation_speeds_clear_floor"] is False
    assert row["details"]["failed_speed_floor_artifacts"] == [
        {
            "artifact": "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-nocache-20260525.json",
            "min_decode_tok_s": 90.0,
            "min_internal_generation_tok_s": 70.0,
            "failed_metrics": ["internal_generation"],
        }
    ]


def test_objective_proof_digest_keeps_cross_family_live_smoke_open_on_non_mimo_gap(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "ZAYA1-8B-MXFP4",
                        "model_type": "zaya",
                        "cache_family": "zaya_cca",
                        "is_mllm": False,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 46,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "blue cat",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "ZAYA1-VL-8B-JANGTQ4",
                        "model_type": "zaya1_vl",
                        "cache_family": "zaya_cca",
                        "is_mllm": True,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [
                                {
                                    "label": "text_cache_repeat_1",
                                    "reason": "expected_ack_missing",
                                }
                            ],
                            "cache_summary": {
                                "has_cache_hit": False,
                                "cache_hit_tokens": 0,
                            },
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [
                                {
                                    "label": "text_cache_repeat_2",
                                    "reason": "expected_ack_missing",
                                }
                            ],
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 40,
                            },
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "Blue",
                        },
                        {
                            "label": "vl_red_image_changed",
                            "validation_failures": [],
                            "content": "Red",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-zaya-vl-jangtq4-true-bundled-toolprobe-media-sentinel-20260525/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "ZAYA1-VL-8B-JANGTQ4",
                        "model_type": "zaya1_vl",
                        "cache_family": "zaya_cca",
                        "is_mllm": True,
                        "supports_video": False,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": False,
                                "cache_hit_tokens": 0,
                            },
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 54,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "blue and cat",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "Blue",
                        },
                        {
                            "label": "text_no_media_after_image",
                            "validation_failures": [],
                            "content": "No.",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-zaya-vl-jangtq4-ack-diagnostic-20260524/external_probe.json",
        {
            "results": [
                {
                    "label": "short_exact_ack",
                    "content": "I'm here to help you with any questions or tasks you have.",
                },
                {
                    "label": "json_ack",
                    "content": '{\n  "answer": "ACK"\n}',
                },
                {
                    "label": "system_cache_repeat_1",
                    "content": "ACK",
                },
                {
                    "label": "system_cache_repeat_2",
                    "content": "ACK",
                    "usage": {
                        "prompt_tokens_details": {
                            "cached_tokens": 54,
                            "cache_detail": "paged+zaya_cca",
                        }
                    },
                },
                {
                    "label": "user_strong_cache_repeat_2",
                    "content": "The output is: ACK",
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-zaya-vl-jangtq4-ack-diagnostic-20260524/rendered_prompt_compare.json",
        {
            "rows": [
                {
                    "name": "zaya_vl_jangtq4",
                    "rendered": "<|im_start|>user\nReply exactly: ACK<|im_end|>\n<|im_start|>assistant\n",
                    "token_count": 12,
                },
                {
                    "name": "zaya_text_mxfp4",
                    "rendered": "<bos><|im_start|>system\n<|im_end|>\n<|im_start|>user\nReply exactly: ACK",
                    "token_count": 22,
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-nemotron-omni-jangtq-bundled-20260524/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        "model_type": "nemotron_h",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": True,
                        "supports_video": True,
                    },
                    "probe_options": {"include_media": True, "include_video": False},
                    "requests": [
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 47,
                            },
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "Blue",
                        },
                        {
                            "label": "text_no_media_after_image",
                            "validation_failures": [],
                            "content": "NONE",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_red_image_changed",
                            "validation_failures": [],
                            "content": "Red",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-nemotron-omni-jangtq-explicit-nomedia-bundled-20260524/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "probe_failed",
                    "row": {
                        "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        "model_type": "nemotron_h",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": True,
                        "supports_video": True,
                        "supports_thinking": True,
                    },
                    "probe_options": {"include_media": True, "include_video": False},
                    "requests": [
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 57,
                            },
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "Blue",
                        },
                        {
                            "label": "text_no_media_after_image",
                            "validation_failures": [
                                {
                                    "label": "text_no_media_after_image",
                                    "reason": "expected_no_media_missing",
                                }
                            ],
                            "content": "Yes",
                        },
                        {
                            "label": "vl_red_image_changed",
                            "validation_failures": [],
                            "content": "Red",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        "model_type": "nemotron_h",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": True,
                        "supports_video": True,
                        "supports_thinking": True,
                    },
                    "probe_options": {"include_media": True, "include_video": False},
                    "requests": [
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 57,
                            },
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "Blue",
                        },
                        {
                            "label": "text_no_media_after_image",
                            "validation_failures": [],
                            "content": "NONE",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_red_image_changed",
                            "validation_failures": [],
                            "content": "Red",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-nemotron-omni-no-media-carryover-diagnostic-20260524b/result.json",
        {
            "results": [
                {
                    "label": "text_no_media_before_any_image",
                    "content": "Yes",
                },
                {
                    "label": "text_no_media_before_any_image_repeat",
                    "content": "Yes",
                },
                {
                    "label": "vl_blue_image",
                    "content": "\nBlue",
                },
                {
                    "label": "text_no_media_after_image",
                    "content": "Yes",
                },
                {
                    "label": "vl_red_image",
                    "content": "\nRed",
                },
                {
                    "label": "text_no_media_after_red_image",
                    "content": "Yes",
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-nemotron-omni-no-media-prompt-variants-20260524/result.json",
        {
            "results": [
                {"label": "ambiguous_before", "content": "Yes"},
                {"label": "negative_control_before", "content": "Yes"},
                {"label": "attachment_none_before", "content": "IMAGE"},
                {"label": "count_before", "content": "1"},
                {"label": "vl_blue_image", "content": "\nBlue"},
                {"label": "ambiguous_after", "content": "Yes"},
                {"label": "negative_control_after", "content": "Yes"},
                {"label": "attachment_none_after", "content": "IMAGE"},
                {"label": "count_after", "content": "1"},
                {"label": "vl_red_image", "content": "\nRed"},
                {"label": "attachment_none_after_red", "content": "IMAGE"},
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-nemotron-omni-no-media-system-negative-diagnostic-20260524/result.json",
        {
            "variants": [
                {"label": "current_no_media_before_any_image", "content": "Yes"},
                {"label": "system_no_media_before_any_image", "content": "NO"},
                {"label": "count_zero_before_any_image", "content": "0"},
                {"label": "forced_output_no_before_any_image", "content": "NO"},
                {"label": "vl_blue_image", "content": "\nBlue"},
                {"label": "current_no_media_after_blue_image", "content": "Yes"},
                {"label": "system_no_media_after_blue_image", "content": "NO"},
                {"label": "count_zero_after_blue_image", "content": "0"},
                {"label": "forced_output_no_after_blue_image", "content": "NO"},
                {"label": "vl_red_image", "content": "\nRed"},
                {"label": "system_no_media_after_red_image", "content": "NO"},
                {"label": "count_zero_after_red_image", "content": "0"},
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-nemotron-omni-no-media-system-prompt-diagnostic-20260524/result.json",
        {
            "results": [
                {"label": "user_only_none_before", "content": "IMAGE"},
                {"label": "system_fact_none_before", "content": "NONE"},
                {"label": "system_exact_none_before", "content": "NONE"},
                {"label": "system_exact_count_before", "content": "1"},
                {"label": "vl_blue_image", "content": "\nBlue"},
                {"label": "system_fact_none_after", "content": "NONE"},
                {"label": "system_exact_none_after", "content": "NONE"},
                {"label": "system_exact_count_after", "content": "1"},
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-video-capfix-bundled-20260526/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                        "model_type": "gemma4",
                        "cache_family": "mixed_attention",
                        "is_mllm": True,
                        "supports_thinking": True,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 56,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "color=blue and animal=cat",
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "Blue",
                        },
                        {
                            "label": "text_no_media_after_image",
                            "validation_failures": [],
                            "content": "NONE",
                        },
                        {
                            "label": "vl_red_image_changed",
                            "validation_failures": [],
                            "content": "Red",
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260525-rerun/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "Ling-2.6-flash-JANGTQ",
                        "model_type": "bailing_hybrid",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": False,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 49,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "You asked me to remember that the color is blue and the animal is cat.",
                            "cache_summary": {"has_cache_hit": True},
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260525-rerun/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "Qwen3.6-27B-MXFP4-CRACK",
                        "model_type": "qwen3_5",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": True,
                        "supports_video": True,
                        "supports_thinking": True,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 41,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "You asked me to remember color=blue and animal=cat.",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_blue_image",
                            "validation_failures": [],
                            "content": "blue",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "text_no_media_after_image",
                            "validation_failures": [],
                            "content": "no",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "vl_blue_video",
                            "validation_failures": [],
                            "content": "blue",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "text_no_media_after_video",
                            "validation_failures": [],
                            "content": "no",
                            "cache_summary": {"has_cache_hit": True},
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-hy3-jangtq2-bundled-toolprobe-20260525/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "Hy3-preview-JANGTQ2",
                        "model_type": "hy_v3",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": False,
                        "supports_video": False,
                        "supports_thinking": True,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 48,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "color=blue and animal=cat",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": True},
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-minimax-small-jangtq-bundled-20260525-rerun/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "MiniMax-M2.7-Small-JANGTQ",
                        "model_type": "minimax_m2",
                        "cache_family": "hybrid_ssm",
                        "is_mllm": False,
                        "supports_video": False,
                        "supports_thinking": True,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 72,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "The user asked me to remember color=blue and animal=cat.",
                            "cache_summary": {"has_cache_hit": True},
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": True},
                        },
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-dsv4-jangtq-k-bundled-cachehit-20260524/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "DeepSeek-V4-Flash-JANGTQ-K",
                        "model_type": "deepseek_v4",
                        "cache_family": "deepseek_v4_composite",
                        "is_mllm": False,
                        "supports_video": False,
                        "supports_thinking": True,
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 3639,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "color=blue and animal=cat",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                    ],
                }
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows[
        "Cross-family live multi-turn smoke matrix is release-cleared"
    ]
    assert row["status"] == "open"
    assert row["details"]["covered_family_keys"] == [
        "dsv4",
        "gemma4",
        "hy3",
        "ling_bailing",
        "minimax",
        "qwen36",
        "zaya_text",
        "zaya_vl",
    ]
    assert row["details"]["missing_required_family_keys"] == ["mimo_v2", "nemotron"]
    assert row["details"]["non_mimo_status"] == "open"
    assert row["details"]["non_mimo_missing_required_family_keys"] == ["nemotron"]
    assert row["details"]["non_mimo_not_pass_artifacts"] == [
        "build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json"
    ]
    assert row["details"]["not_pass_required_family_artifacts"] == {
        "nemotron": [
            "build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json"
        ],
    }
    assert row["details"]["blocking_required_family_artifacts"] == {
        "mimo_v2": [
            "build/current-all-local-model-smoke-mimo-v2-jang2l-bundled-20260527/summary.json"
        ],
        "nemotron": [
            "build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun/summary.json"
        ],
    }


def test_objective_proof_digest_accepts_cross_family_live_smoke_when_only_mimo_is_deferred(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import (
        ALL_LOCAL_MODEL_SMOKE_ARTIFACTS_BY_FAMILY,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    family_rows = {
        "dsv4": ("DeepSeek-V4-Flash-JANGTQ-K", "deepseek_v4"),
        "gemma4": ("Gemma-4-26B-A4B-JANG_4M-CRACK", "gemma4"),
        "hy3": ("Hy3-JANGTQ2", "hy3"),
        "ling_bailing": ("Ling-Bailing-JANGTQ", "ling"),
        "minimax": ("MiniMax-Small-JANGTQ", "minimax"),
        "nemotron": ("Nemotron-Omni-Nano-JANGTQ-CRACK", "nemotron_h"),
        "qwen36": ("Qwen3.6-27B-MXFP4-CRACK", "qwen3_5"),
        "zaya_text": ("ZAYA1-8B-MXFP4", "zaya"),
        "zaya_vl": ("ZAYA1-VL-8B-JANGTQ4", "zaya1_vl"),
    }
    reasoning_families = {"dsv4", "gemma4", "hy3", "minimax", "qwen36"}
    for family, (name, model_type) in family_rows.items():
        requests = [
            {
                "label": "text_cache_repeat_2",
                "validation_failures": [],
                "content": "ACK",
                "cache_summary": {
                    "has_cache_hit": True,
                    "cache_hit_tokens": 48,
                },
            }
        ]
        if family in reasoning_families:
            requests.append(
                {
                    "label": "reasoning_on",
                    "validation_failures": [],
                    "content": "FINAL=OK",
                    "reasoning_chars": 24,
                    "cache_summary": {"has_cache_hit": True},
                }
            )
        _write_json(
            tmp_path,
            ALL_LOCAL_MODEL_SMOKE_ARTIFACTS_BY_FAMILY[family][0],
            {
                "status": "pass",
                "completed": 1,
                "row_count": 1,
                "results": [
                    {
                        "status": "pass",
                        "row": {
                            "name": name,
                            "model_type": model_type,
                            "cache_family": f"{family}_cache",
                        },
                        "requests": requests,
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-mimo-v2-jang2l-bundled-20260527/summary.json",
        {
            "status": "fail",
            "completed": 1,
            "failed": 1,
            "results": [
                {
                    "status": "fail",
                    "row": {
                        "name": "MiMo-V2-Flash-JANG_2L",
                        "model_type": "mimo_v2",
                        "cache_family": "mimo_v2_hybrid_swa",
                    },
                    "failures": ["mimo_deferred_broken_model"],
                    "requests": [
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "incoherent",
                            "cache_summary": {
                                "has_cache_hit": True,
                                "cache_hit_tokens": 48,
                            },
                        },
                        {
                            "label": "text_multiturn_recall",
                            "validation_failures": [],
                            "content": "incoherent",
                        },
                    ],
                }
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows["Cross-family live multi-turn smoke matrix is release-cleared"]

    assert row["status"] == "pass"
    assert row["details"]["non_mimo_status"] == "pass"
    assert row["details"]["release_boundary"] == (
        "non_mimo_live_smoke_clear_mimo_v2_deferred"
    )
    assert row["details"]["mimo_v2_deferred"] is True
    assert row["details"]["not_pass_required_family_artifacts"] == {
        "mimo_v2": [
            "build/current-all-local-model-smoke-mimo-v2-jang2l-bundled-20260527/summary.json"
        ],
    }
    assert row["details"]["missing_required_family_keys"] == ["mimo_v2"]
    assert row["details"]["non_mimo_missing_required_family_keys"] == []
    assert row["details"]["non_mimo_not_pass_artifacts"] == []


def test_all_local_smoke_digest_revalidates_cjk_visible_text_from_stale_artifacts(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import _all_local_model_smoke_detail

    rel = "build/current-all-local-model-smoke-zaya-text-bundled-20260524/summary.json"
    _write_json(tmp_path, rel, {"present": True})
    ok, details = _all_local_model_smoke_detail(
        [
            (
                rel,
                {
                    "completed": 1,
                    "row_count": 1,
                    "results": [
                        {
                            "status": "pass",
                            "row": {
                                "name": "ZAYA1-8B-MXFP4",
                                "model_type": "zaya",
                                "cache_family": "zaya_cca",
                                "is_mllm": False,
                            },
                            "requests": [
                                {
                                    "label": "text_cache_repeat_1",
                                    "validation_failures": [],
                                    "content": "ACK",
                                    "cache_summary": {"has_cache_hit": False},
                                },
                                {
                                    "label": "text_cache_repeat_2",
                                    "validation_failures": [],
                                    "content": "ACK",
                                    "cache_summary": {
                                        "has_cache_hit": True,
                                        "cache_hit_tokens": 40,
                                    },
                                },
                                {
                                    "label": "text_multiturn_recall",
                                    "validation_failures": [],
                                    "content": "blue cat 蓝色 蓝色",
                                    "cache_summary": {"has_cache_hit": True},
                                },
                            ],
                        }
                    ],
                },
            )
        ],
        tmp_path,
    )

    artifact = details["artifacts"][0]
    assert ok is False
    assert artifact["status"] == "open"
    assert {
        "label": "text_multiturn_recall",
        "reason": "unexpected_cjk_visible_text",
        "cjk_chars": 4,
    } in artifact["validation_failures"]


def test_objective_digest_includes_current_real_ui_unblocked_non_mimo_matrix(
    tmp_path,
):
    from tests.cross_matrix import summarize_objective_proof as objective

    _write_json(
        tmp_path,
        objective.CURRENT_RELEASE_REGRESSION_MANIFEST_REL,
        {
            "current_proof_sweep": {
                "real_ui_live_model_matrix": {
                    "status": "open",
                    "release_blocker": (
                        "Real Electron UI cross-family live model matrix is release-cleared"
                    ),
                    "missing_families": ["mimo_v2", "dsv4"],
                    "partial_families": ["mimo_v2", "dsv4"],
                    "unblocked_non_mimo_status": "pass",
                    "unblocked_non_mimo_missing_families": [],
                    "unblocked_non_mimo_partial_families": [],
                    "unblocked_non_mimo_excluded_families": ["dsv4", "mimo_v2"],
                    "resource_blockers": {
                        "dsv4": {
                            "reason": "insufficient_memory",
                            "artifact": (
                                "build/current-real-ui-dsv4-memory-preflight-20260530-local-refresh.json"
                            ),
                        }
                    },
                    "covered_families": {
                        "zaya_text": {
                            "status": "pass",
                            "artifact": (
                                "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-runcommand-tools-20260527-proof.json"
                            ),
                            "covered_surfaces": [
                                "chat_completions",
                                "responses_api",
                                "long_tool_loop",
                            ],
                            "required_surfaces": [
                                "chat_completions",
                                "responses_api",
                                "long_tool_loop",
                            ],
                        },
                        "gemma4": {
                            "status": "pass",
                            "artifact": (
                                "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-20260527-proof.json"
                            ),
                            "covered_surfaces": [
                                "chat_completions",
                                "responses_api",
                                "reasoning_display",
                            ],
                            "required_surfaces": [
                                "chat_completions",
                                "responses_api",
                                "reasoning_display",
                            ],
                        },
                    },
                }
            }
        },
    )
    _write_json(
        tmp_path,
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-runcommand-tools-20260527-proof.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path,
        "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-20260527-proof.json",
        {"status": "pass"},
    )

    digest = objective.build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows["Real Electron UI cross-family live model matrix is release-cleared"]

    assert row["status"] == "open"
    assert row["details"]["real_ui_live_model_matrix"]["status"] == "open"
    assert (
        row["details"]["real_ui_live_model_matrix"]["unblocked_non_mimo_status"]
        == "pass"
    )
    assert row["details"]["real_ui_live_model_matrix"]["missing_families"] == [
        "mimo_v2",
        "dsv4",
    ]
    assert row["details"]["real_ui_missing_required_family_keys"] == [
        "mimo_v2",
        "dsv4",
    ]
    assert row["details"]["real_ui_blocking_required_family_artifacts"] == {
        "dsv4": [
            "build/current-real-ui-dsv4-memory-preflight-20260530-local-refresh.json"
        ],
        "mimo_v2": [
            "build/current-all-local-model-smoke-mimo-v2-jang2l-bundled-20260527/summary.json"
        ],
    }
    assert row["details"]["real_ui_live_model_matrix"][
        "unblocked_non_mimo_excluded_families"
    ] == ["dsv4", "mimo_v2"]
    assert row["details"]["real_ui_live_model_matrix"]["covered_families"][
        "gemma4"
    ]["status"] == "pass"
    assert row["details"]["release_boundary"] == (
        "current_real_ui_matrix_is_authoritative_for_unblocked_non_mimo"
    )

    non_mimo_row = rows[
        "Real Electron UI unblocked non-MiMo live model matrix is proven"
    ]
    assert non_mimo_row["status"] == "pass"
    assert non_mimo_row["details"]["release_boundary"] == (
        "unblocked_non_mimo_real_ui_matrix_proven_with_explicit_exclusions"
    )
    assert non_mimo_row["details"]["excluded_families"] == ["dsv4", "mimo_v2"]
    assert non_mimo_row["details"]["covered_family_keys"] == [
        "gemma4",
        "zaya_text",
    ]


def test_objective_digest_keeps_unblocked_non_mimo_real_ui_open_on_missing_family(
    tmp_path,
):
    from tests.cross_matrix import summarize_objective_proof as objective

    _write_json(
        tmp_path,
        objective.CURRENT_RELEASE_REGRESSION_MANIFEST_REL,
        {
            "current_proof_sweep": {
                "real_ui_live_model_matrix": {
                    "status": "open",
                    "release_blocker": (
                        "Real Electron UI cross-family live model matrix is release-cleared"
                    ),
                    "missing_families": ["mimo_v2", "dsv4", "qwen36"],
                    "partial_families": ["mimo_v2", "dsv4"],
                    "unblocked_non_mimo_status": "open",
                    "unblocked_non_mimo_missing_families": ["qwen36"],
                    "unblocked_non_mimo_partial_families": [],
                    "unblocked_non_mimo_excluded_families": ["dsv4", "mimo_v2"],
                    "resource_blockers": {
                        "dsv4": {"reason": "insufficient_memory"}
                    },
                    "covered_families": {
                        "zaya_text": {
                            "status": "pass",
                            "artifact": (
                                "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-runcommand-tools-20260527-proof.json"
                            ),
                            "covered_surfaces": ["chat_completions"],
                            "required_surfaces": ["chat_completions"],
                        },
                    },
                }
            }
        },
    )

    digest = objective.build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows["Real Electron UI unblocked non-MiMo live model matrix is proven"]

    assert row["status"] == "open"
    assert row["details"]["missing_family_keys"] == ["qwen36"]
    assert row["details"]["partial_family_keys"] == []


def test_objective_digest_overrides_stale_real_ui_dsv4_resource_blocker(
    tmp_path,
):
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import summarize_objective_proof as objective

    stale_artifact = (
        "build/current-real-ui-dsv4-memory-preflight-20260528-signing-ledger-refresh.json"
    )
    current_artifact = manifest.CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    _write_json(
        tmp_path,
        objective.CURRENT_RELEASE_REGRESSION_MANIFEST_REL,
        {
            "current_proof_sweep": {
                "real_ui_live_model_matrix": {
                    "status": "open",
                    "missing_families": ["dsv4"],
                    "partial_families": ["dsv4"],
                    "resource_blockers": {
                        "dsv4": {
                            "reason": "insufficient_memory",
                            "artifact": stale_artifact,
                            "memory_gap_gb": 49.23,
                        }
                    },
                }
            }
        },
    )
    _write_json(
        tmp_path,
        current_artifact,
        {
            "status": "skipped_insufficient_memory",
            "launch_decision": "do_not_launch",
            "did_not_launch": True,
            "model_path": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
            "required_available_gb": 120.0,
            "free_plus_speculative_purgeable_gb": 58.64,
            "memory_gap_gb": 61.36,
        },
    )

    digest = objective.build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows["Real Electron UI cross-family live model matrix is release-cleared"]
    dsv4_blocker = row["details"]["real_ui_live_model_matrix"]["resource_blockers"][
        "dsv4"
    ]

    assert dsv4_blocker["artifact"] == current_artifact
    assert dsv4_blocker["memory_gap_gb"] == 61.36
    assert stale_artifact not in json.dumps(digest)


def test_all_local_smoke_digest_revalidates_wider_cjk_kana_and_hangul_ranges(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import _all_local_model_smoke_detail

    rel = "build/current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json"
    _write_json(tmp_path, rel, {"present": True})
    ok, details = _all_local_model_smoke_detail(
        [
            (
                rel,
                {
                    "completed": 1,
                    "row_count": 1,
                    "results": [
                        {
                            "status": "pass",
                            "row": {
                                "name": "ZAYA1-VL-8B-JANGTQ4",
                                "model_type": "zaya1_vl",
                                "cache_family": "zaya_cca",
                                "is_mllm": True,
                            },
                            "requests": [
                                {
                                    "label": "vl_blue_image",
                                    "validation_failures": [],
                                    "content": "blue ｶﾀｶﾅ 한 𠀋",
                                    "cache_summary": {"has_cache_hit": True},
                                },
                            ],
                        }
                    ],
                },
            )
        ],
        tmp_path,
    )

    artifact = details["artifacts"][0]
    assert ok is False
    assert artifact["status"] == "open"
    assert {
        "label": "vl_blue_image",
        "reason": "unexpected_cjk_visible_text",
        "cjk_chars": 8,
    } in artifact["validation_failures"]


def test_all_local_smoke_digest_revalidates_reasoning_loop_from_stale_artifacts(
    tmp_path,
):
    from tests.cross_matrix.summarize_objective_proof import _all_local_model_smoke_detail

    rel = "build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260524/summary.json"
    _write_json(tmp_path, rel, {"present": True})
    ok, details = _all_local_model_smoke_detail(
        [
            (
                rel,
                {
                    "completed": 1,
                    "row_count": 1,
                    "results": [
                        {
                            "status": "pass",
                            "row": {
                                "name": "Qwen3.6-27B-MXFP4-CRACK",
                                "model_type": "qwen36",
                            },
                            "requests": [
                                {
                                    "label": "reasoning_on",
                                    "validation_failures": [],
                                    "content": "FINAL=OK",
                                    "reasoning_chars": 5001,
                                    "cache_summary": {"has_cache_hit": True},
                                },
                            ],
                        }
                    ],
                },
            )
        ],
        tmp_path,
    )

    artifact = details["artifacts"][0]
    assert ok is False
    assert artifact["status"] == "open"
    assert {
        "label": "reasoning_on",
        "reason": "reasoning_loop_too_long",
        "reasoning_chars": 5001,
        "max_reasoning_chars": 4096,
    } in artifact["validation_failures"]


def test_objective_proof_digest_requires_dsv4_smoke_cache_hit(tmp_path):
    from tests.cross_matrix.summarize_objective_proof import build_digest

    _write_passing_base_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "build/current-all-local-model-smoke-dsv4-jangtq-k-bundled-cachehit-20260524/summary.json",
        {
            "completed": 1,
            "row_count": 1,
            "results": [
                {
                    "status": "pass",
                    "row": {
                        "name": "DeepSeek-V4-Flash-JANGTQ-K",
                        "model_type": "deepseek_v4",
                        "cache_family": "deepseek_v4_composite",
                    },
                    "requests": [
                        {
                            "label": "text_cache_repeat_1",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "text_cache_repeat_2",
                            "validation_failures": [],
                            "content": "ACK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                        {
                            "label": "reasoning_on",
                            "validation_failures": [],
                            "content": "FINAL=OK",
                            "cache_summary": {"has_cache_hit": False},
                        },
                    ],
                }
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    row = rows["Cross-family live multi-turn smoke matrix is release-cleared"]

    assert row["status"] == "open"
    assert "dsv4" in row["details"]["missing_cache_hit_family_keys"]
    dsv4 = next(
        artifact for artifact in row["details"]["artifacts"] if artifact["family_keys"] == ["dsv4"]
    )
    assert dsv4["family_keys"] == ["dsv4"]
    assert dsv4["cache_hit_observed"] is False
    assert dsv4["missing_cache_hit_family_keys"] == ["dsv4"]


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
    from tests.cross_matrix.summarize_objective_proof import API_CACHE_CONTRACT_REL, build_digest

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / API_CACHE_CONTRACT_REL).unlink()

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Current-source API adapters and non-DSV4 cache contracts are no-heavy covered"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [API_CACHE_CONTRACT_REL]


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
    from tests.cross_matrix.summarize_objective_proof import (
        MAX_OUTPUT_CONTEXT_CONTRACT_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / MAX_OUTPUT_CONTEXT_CONTRACT_REL).unlink()

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Server default max output and max context are distinct and map to correct CLI flags"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [MAX_OUTPUT_CONTEXT_CONTRACT_REL]


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
    from tests.cross_matrix.summarize_objective_proof import (
        PARSER_REGISTRY_CONTRACT_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / PARSER_REGISTRY_CONTRACT_REL).unlink(missing_ok=True)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["High-risk model family parser, artifact, and launch policy gates are current"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [PARSER_REGISTRY_CONTRACT_REL]


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
    from tests.cross_matrix.summarize_objective_proof import (
        NATIVE_MTP_CONTRACT_REL,
        build_digest,
    )

    _write_passing_base_artifacts(tmp_path)
    (tmp_path / NATIVE_MTP_CONTRACT_REL).unlink(missing_ok=True)

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    row = rows["Generation defaults, Native MTP, and VL media gates are current"]
    assert row["status"] == "open"
    assert row["details"]["missing_evidence"] == [NATIVE_MTP_CONTRACT_REL]


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
    from tests.cross_matrix.summarize_objective_proof import API_CACHE_CONTRACT_REL, build_digest

    _write_passing_base_artifacts(tmp_path)
    contract_path = tmp_path / API_CACHE_CONTRACT_REL
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
        "build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json",
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
        "build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json",
        {
            "status": "pass",
            "cases": [
                {
                    "name": "chatmax_original",
                    "exact": True,
                    "normalized_exact": True,
                    "content": "const camera = new THREE.PerspectiveCamera();\nconst geometry = new THREE.BoxGeometry();\nconst material = new THREE.MeshBasicMaterial();",
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": False,
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json",
        {
            "status": "pass",
            "cases": [
                {
                    "name": "chatmax_1024_budget",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "finish_reason": "stop",
                    "missing": [],
                    "corrupt_patterns": [],
                    "completion_tokens": 195,
                    "content": "const camera = new THREE.PerspectiveCamera();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "dsv4_policy_reason": "dsv4_direct_rail_identifier_unsafe",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json",
        {"status": "pass", "cases": []},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json",
        {"status": "pass", "cases": []},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json",
        {"status": "pass", "cases": []},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-scene-token-rank-contrast-live-20260524-1324.json",
        {"status": "pass", "cases": []},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260524.json",
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
        "build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json",
        {
            "status": "pass",
            "health_before": {"scheduler": {"cache_hit_tokens": 0}},
            "health_after": {"scheduler": {"cache_hit_tokens": 0}},
            "diagnostic_summary": {
                "colon_control_exact": True,
                "period_control_exact": True,
                "system_no_draft_exact": True,
                "system_verify_exact": True,
            },
            "cases": [
                {
                    "name": "colon_known_pass_control",
                    "exact": True,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "hidden_contains_corruption": False,
                    "corrupt_patterns": [],
                    "missing": [],
                },
                {
                    "name": "period_known_fail_control",
                    "exact": True,
                    "prompt_tokens": 90,
                    "completion_tokens": 195,
                    "hidden_contains_corruption": False,
                    "corrupt_patterns": [],
                    "missing": [],
                },
                {
                    "name": "period_no_reasoning_code_draft_system",
                    "exact": True,
                    "prompt_tokens": 130,
                    "completion_tokens": 144,
                    "hidden_contains_corruption": False,
                    "corrupt_patterns": [],
                    "missing": [],
                },
                {
                    "name": "period_verify_identifiers_system",
                    "exact": True,
                    "prompt_tokens": 140,
                    "completion_tokens": 195,
                    "hidden_contains_corruption": False,
                    "corrupt_patterns": [],
                    "missing": [],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-template-parity-diagnostic-20260524-1343.json",
        {
            "schema": "vmlx-dsv4-template-parity-diagnostic-v1",
            "all_sidecar_equals_tokenizer_template": True,
            "cases": [
                {
                    "name": "period_exact_code",
                    "enable_thinking": False,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_closed",
                    "boundary_tokens_tail": [".Ċ", "const"],
                },
                {
                    "name": "colon_exact_code",
                    "enable_thinking": False,
                    "reasoning_effort": None,
                    "sidecar_equals_tokenizer_template": True,
                    "prompt_tokens": 90,
                    "assistant_suffix": "thinking_closed",
                    "boundary_tokens_tail": [":Ċ", "const"],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-prefill-execution-variant-logits-20260524.json",
        {
            "schema": "vmlx-dsv4-prefill-execution-variant-logits-v1",
            "variants": [],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-prompt-variant-logit-probe-20260524.json",
        {
            "schema": "vmlx-dsv4-prompt-variant-logit-probe-v1",
            "cases": [],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-reasoning-policy-live-20260524-1408.json",
        {
            "schema": "vmlx-dsv4-reasoning-policy-live-v1",
            "status": "pass",
            "cases": [
                {
                    "name": "one_token_chat_explicit_off",
                    "content_is_null": False,
                    "reasoning_chars": 0,
                },
                {
                    "name": "short_chat_explicit_off",
                    "content_is_null": False,
                    "reasoning_chars": 0,
                },
                {
                    "name": "short_chat_requested_on",
                    "content_is_null": True,
                    "reasoning_chars": 36,
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json",
        {"schema": "vmlx-dsv4-cache-vs-full-logit-isolation-v1", "cases": []},
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json",
        {"schema": "vmlx-dsv4-batch-generator-logit-trace-v1", "cases": []},
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
                "identifier_gate": "build/current-dsv4-identifier-count-ablation-20260521/result.json",
                "full_output_gate": "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
        {
            "status": "pass",
            "reason": "sufficient_free_memory",
            "required_available_gb": 120.0,
            "model": "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
            "selected_cases": [
                "chat_off_rep1",
                "chat_off_no_punct_rep1",
                "responses_off_rep1",
                "responses_off_no_punct_rep1",
            ],
            "case_count": 4,
            "telemetry": [
                {
                    "name": "preflight",
                    "system_memory": {"available_gb": 192.0, "total_gb": 256.0},
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "pass",
            "selected_cases": [
                "chat_off",
                "responses_off",
                "legacy_completion_raw",
            ],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_not_requested",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-bundle-defaults-source-20260525.json",
        {
            "status": "pass",
            "selected_cases": [
                "chat_off_bundle_defaults",
                "responses_off_bundle_defaults",
            ],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_bundle_defaults",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "content": "const renderer = new THREE.WebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off_bundle_defaults",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "content": "const renderer = new THREE.WebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-bundle-defaults-dryrun-20260525.json",
        {
            "status": "dry_run",
            "selected_cases": [
                "chat_off_bundle_defaults",
                "responses_off_bundle_defaults",
            ],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_bundle_defaults",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
                {
                    "name": "responses_off_bundle_defaults",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json",
        {
            "status": "pass",
            "selected_cases": ["chat_off_rep1", "responses_off_rep1"],
            "case_count": 2,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "content": "const renderer = new THREE.WebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off_rep1",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "content": "const renderer = new THREE.WebGLRenderer();",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json",
        {
            "status": "dry_run",
            "selected_cases": [
                "chat_off_rep1",
                "chat_on_rep1",
                "responses_off_rep1",
                "responses_on_rep1",
            ],
            "case_count": 4,
            "cases": [
                {
                    "name": "chat_off_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
                {
                    "name": "chat_on_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "request_overrides": {"enable_thinking": True},
                },
                {
                    "name": "responses_off_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                    },
                    "request_overrides": {"enable_thinking": False},
                },
                {
                    "name": "responses_on_rep1",
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "requested_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                    },
                    "request_overrides": {"enable_thinking": True},
                },
            ],
        },
    )
    for rel in (
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json",
        "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json",
    ):
        _write_json(
            tmp_path,
            rel,
            {
                "status": "dry_run",
                "selected_cases": [
                    "chat_off_rep1",
                    "chat_off_no_punct_rep1",
                    "responses_off_rep1",
                    "responses_off_no_punct_rep1",
                ],
                "case_count": 4,
                "cases": [],
            },
        )
    for rel in (
        "build/current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json",
        "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
        "build/current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json",
        "build/current-dsv4-route-mode-code-exactness-source-thinking-ab-prefill-logits-eval-20260525.json",
        "build/current-dsv4-route-mode-code-exactness-token-tail-source-ab-20260525.json",
        "build/current-dsv4-route-mode-code-exactness-source-rep1-prefill-logits-eval-20260525.json",
    ):
        _write_json(
            tmp_path,
            rel,
            {
                "status": "pass",
                "cases": [
                    {
                        "name": "chat_on",
                        "route": "chat",
                        "exact": True,
                        "normalized_exact": True,
                        "missing": [],
                        "corrupt_patterns": [],
                        "effective_prompt_diagnostics": {
                            "assistant_suffix_kind": "thinking_open",
                            "dsv4_policy_reason": "requested_thinking",
                        },
                    }
                ],
            },
        )
    _write_json(
        tmp_path,
        "build/current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json",
        {
            "schema": "vmlx-dsv4-direct-vs-thinking-webgl-logit-probe-v1",
            "cases": [
                {
                    "rail": "direct_off",
                    "prefix_name": "after_const_renderer_web",
                    "ranks": {"GL": 1, "Web": 5},
                    "top": [{"token": "GL"}],
                },
                {
                    "rail": "requested_thinking",
                    "prefix_name": "after_const_renderer_web",
                    "ranks": {"GL": 1, "Web": 5},
                    "top": [{"token": "GL"}],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-jang-batch-generator-warmup-ablation-20260524.json",
        {
            "schema": "vmlx-dsv4-batch-generator-warmup-ablation-v1",
            "runs": [
                {
                    "label": "normal_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                },
                {
                    "label": "skip_warmup",
                    "decoded": "THREE.PerspectiveCamera<｜end▁of▁sentence｜>",
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    assert quality["status"] == "pass", quality["details"]
    assert quality["details"]["clearance_checks"]["identifier_integrity"] is True
    open_requirements = [
        item["requirement"] for item in digest["requirements"] if item["status"] == "open"
    ]
    assert open_requirements == [
        "Ling/Bailing multilingual output quality is release-cleared",
        "Gemma4 26B CRACK mixed-SWA app-engine speed floor is release-cleared",
        "Cross-family live multi-turn smoke matrix is release-cleared",
        "Real Electron UI cross-family live model matrix is release-cleared",
    ]

    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "fail",
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": False,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "has_markdown_fence": True,
                    "prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_open",
                        "prompt_endswith_assistant_think_open": True,
                    },
                },
            ],
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    assert quality["status"] == "open"
    assert (
        quality["details"]["current_installed_prompt_rail_exactness_probe"]["status"]
        == "fail"
    )
    assert quality["details"]["failed_quality_gates"] == [
        {
            "gate": "current_installed_prompt_rail_exactness_probe",
            "artifact": "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
            "status": "fail",
            "failed_cases": ["chat_off"],
        },
        {
            "gate": "long_context",
            "status": "review",
        },
    ]


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
    _write_json(
        tmp_path,
        "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
        {
            "status": "pass",
            "selected_cases": [
                "chat_off",
                "responses_off",
                "legacy_completion_raw",
            ],
            "case_count": 3,
            "cases": [
                {
                    "name": "chat_off",
                    "route": "chat",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "responses_off",
                    "route": "responses",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_disabled",
                    },
                },
                {
                    "name": "legacy_completion_raw",
                    "route": "completion",
                    "exact": True,
                    "normalized_exact": True,
                    "missing": [],
                    "corrupt_patterns": [],
                    "effective_prompt_diagnostics": {
                        "assistant_suffix_kind": "thinking_closed",
                        "dsv4_policy_reason": "thinking_not_requested",
                    },
                },
            ],
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
    assert quality["details"]["legacy_clearance_artifacts"] == {
        "identifier_gate": "build/dsv4-source-identifier/result.json",
        "full_output_gate": "build/dsv4-source-full-output/result.json",
    }
    assert quality["details"]["missing_clearance_artifacts"] == []
    assert quality["details"]["non_current_clearance_reason"] == (
        "Legacy DSV4 clearance artifacts are not accepted as current release evidence."
    )


def test_objective_proof_digest_reports_missing_current_dsv4_clearance_artifacts(
    tmp_path,
):
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
                "identifier_gate": "build/current-dsv4-identifier-count-ablation-20260521/result.json",
                "full_output_gate": "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
            },
        },
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}

    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]
    assert quality["status"] == "open"
    assert quality["details"]["legacy_clearance_artifacts"] == {}
    assert quality["details"]["missing_clearance_artifacts"] == [
        "build/current-dsv4-identifier-count-ablation-20260521/result.json",
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260530-local-refresh.json",
    ]


def test_objective_proof_digest_classifies_legacy_dsv4_clearance_artifacts_as_stale(
    tmp_path,
):
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
                "identifier_gate": "build/current-dsv4-identifier-count-ablation-20260521/result.json",
                "full_output_gate": "build/dsv4-source-full-output/result.json",
                "prompt_rail_ablation": "build/dsv4-chat-prompt-ablation-20260520101331/result.json",
            },
        },
    )
    _write_json(
        tmp_path,
        "build/current-dsv4-identifier-count-ablation-20260521/result.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path,
        "build/dsv4-source-full-output/result.json",
        {"status": "pass"},
    )
    _write_json(
        tmp_path,
        "build/dsv4-chat-prompt-ablation-20260520101331/result.json",
        {"status": "pass"},
    )

    digest = build_digest(tmp_path)
    rows = {item["requirement"]: item for item in digest["requirements"]}
    quality = rows["DSV4 long-output/code/file-generation quality is release-cleared"]

    assert quality["status"] == "open"
    assert quality["details"]["legacy_clearance_artifacts"] == {
        "full_output_gate": "build/dsv4-source-full-output/result.json",
        "prompt_rail_ablation": "build/dsv4-chat-prompt-ablation-20260520101331/result.json",
    }
    assert "build/dsv4-source-full-output/result.json" not in quality["details"][
        "missing_clearance_artifacts"
    ]
    assert (
        "build/dsv4-chat-prompt-ablation-20260520101331/result.json"
        not in quality["details"]["missing_clearance_artifacts"]
    )

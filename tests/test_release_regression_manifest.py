import json
import re
from pathlib import Path
import shlex
import hashlib

from tests.cross_matrix import release_regression_manifest
from tests.cross_matrix.run_release_regression_manifest import build_manifest_artifact
from tests.cross_matrix.release_regression_manifest import (
    CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS,
    CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS,
    CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS,
    CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS,
    CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS,
    CURRENT_DEV_UI_PROOF_ARTIFACTS,
    CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS,
    CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
    CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT,
    CURRENT_POST_BUDGET_EDGE_ARTIFACTS,
    CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT,
    CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT,
    CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
    CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
    CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS,
    CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS,
    CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT,
    CURRENT_REGRESSION_SUITE_ARTIFACT,
    CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
    CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
    EXPECTED_CURRENT_API_SURFACE_CHECKS,
    EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS,
    EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS,
    EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS,
    EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS,
    EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS,
    EXPECTED_CURRENT_MCP_POLICY_CHECKS,
    EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS,
    EXPECTED_CURRENT_NATIVE_MTP_CHECKS,
    EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS,
    EXPECTED_CURRENT_OPEN_REQUIREMENTS,
    EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS,
    DEFERRED_RELEASE_OPEN_REQUIREMENTS,
    EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
    EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS,
    EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS,
    EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS,
    EXPECTED_CURRENT_TOOL_CALL_CHECKS,
    EXPECTED_CURRENT_VL_MEDIA_CHECKS,
    REQUIRED_RELEASE_DOMAINS,
    REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES,
    REQUIRED_REAL_UI_LIVE_MODEL_SURFACES,
    REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY,
    _required_live_smoke_request_labels,
    _current_regression_suite_state_is_acceptable,
    _annotate_real_ui_unblocked_non_mimo_status,
    _current_release_blocker_ledger,
    _live_smoke_gap_is_expected_mimo_open,
    _real_ui_architecture_cache_policy_ok,
    _real_ui_named_tool_probe_semantics_ok,
    _validate_current_issue175_179_release_boundary_audit,
    _validate_current_issue181_183_runtime_audit,
    _validate_current_public_app_issue_audit,
    _validate_current_release_surface_matrix_artifact,
    _validate_current_real_ui_live_model_matrix,
    _validate_current_step37_vlm_runtime_audit,
    build_manifest,
    validate_current_dsv4_proof_artifact_freshness,
    validate_current_proof_sweep_artifacts,
)

_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _passing_cache_family_matrix():
    return {
        row: {
            "status": "pass",
            "checks": {},
            "missing_markers": [],
            "missing_api_checks": [],
            "missing_api_command_markers": [],
            "missing_panel_markers": [],
        }
        for row in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
    }


def _passing_cache_architecture_payload():
    return {
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS},
        "missing_markers": [],
        "missing_api_checks": [],
        "missing_api_command_markers": [],
        "missing_panel_markers": [],
        "cache_family_matrix": _passing_cache_family_matrix(),
    }


def _passing_generation_defaults_family_matrix():
    return {
        row: {
            "status": "pass",
            "checks": {},
            "missing_markers": [],
        }
        for row in EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS
    }


def _passing_generation_defaults_payload():
    return {
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS},
        "missing_markers": [],
        "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
    }


def test_release_regression_manifest_tracks_n2_pro_397b_known_open_requirement():
    assert (
        "N2 Pro 397B JANG1L/JANGTQ runtime/cache/API/UI quality is release-cleared"
        in EXPECTED_CURRENT_OPEN_REQUIREMENTS
    )


def _write_current_objective_digest(
    root: Path,
    *,
    open_requirements: list[str] | None = None,
    missing_evidence: list[str] | None = None,
) -> None:
    artifact = root / CURRENT_OBJECTIVE_DIGEST_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    open_rows = (
        EXPECTED_CURRENT_OPEN_REQUIREMENTS
        if open_requirements is None
        else open_requirements
    )
    requirements = [
        {
            "requirement": "Cache architecture matrix is release-proven",
            "status": "pass",
            "details": {
                "evidence_files_present": True,
                "missing_evidence": [],
            },
        },
    ]
    for requirement in open_rows:
        details: dict[str, object] = {
            "evidence_files_present": not missing_evidence,
            "missing_evidence": missing_evidence or [],
        }
        if (
            requirement
            == "MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared"
        ):
            details.update(
                {
                    "release_blocker_id": "issue179_minimax_k_root_cause_audit",
                    "not_proven": [
                        "reporter installed app bundle hash matches public/local server.py route proof"
                    ],
                    "release_boundary": (
                        "#179 remains open: reporter bundle/server parity is not proven."
                    ),
                }
            )
        requirements.append(
            {
                "requirement": requirement,
                "status": "open",
                "details": details,
            }
        )
    artifact.write_text(
        json.dumps(
            {
                "requirements": requirements,
                "dsv4_proof_artifact_freshness_evidence": list(
                    CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_current_proof_artifacts_do_not_reference_stale_dsv4_second_local_check():
    freshness = validate_current_dsv4_proof_artifact_freshness(Path("."))

    assert freshness["status"] == "pass"
    assert freshness["checked_artifacts"] == list(
        CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS
    )
    assert freshness["stale_markers"] == ["second-local-check"]
    assert freshness["required_artifacts"] == [
        CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
        CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
    ]
    for artifact in freshness["artifacts"]:
        assert artifact["missing"] is False
        assert artifact["stale_references"] == []
        assert artifact["missing_required_artifacts"] == []


def test_current_dsv4_proof_artifact_freshness_checks_all_current_proof_artifacts():
    assert CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS == (
        CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT,
        CURRENT_REGRESSION_SUITE_ARTIFACT,
        CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
    )


def test_current_dsv4_proof_artifact_freshness_rejects_stale_second_local_check(
    tmp_path,
):
    for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS:
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS) + "\n",
            encoding="utf-8",
        )
    stale_path = tmp_path / CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS[0]
    stale_path.write_text(
        stale_path.read_text(encoding="utf-8") + "second-local-check\n",
        encoding="utf-8",
    )

    freshness = validate_current_dsv4_proof_artifact_freshness(tmp_path)

    assert freshness["status"] == "fail"
    assert freshness["failures"] == [
        {
            "path": CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS[0],
            "missing": False,
            "stale_references": ["second-local-check"],
            "missing_required_artifacts": [],
        }
    ]


def test_current_dsv4_proof_artifact_freshness_rejects_missing_third_local_check(
    tmp_path,
):
    for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS:
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS[0] + "\n",
            encoding="utf-8",
        )

    freshness = validate_current_dsv4_proof_artifact_freshness(tmp_path)

    assert freshness["status"] == "fail"
    assert freshness["failures"] == [
        {
            "path": artifact,
            "missing": False,
            "stale_references": [],
            "missing_required_artifacts": [
                CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS[1]
            ],
        }
        for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS
    ]


def test_current_dsv4_proof_artifact_freshness_rejects_missing_current_artifact(
    tmp_path,
):
    missing_artifact = CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS[0]
    for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS[1:]:
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS) + "\n",
            encoding="utf-8",
        )

    freshness = validate_current_dsv4_proof_artifact_freshness(tmp_path)

    assert freshness["status"] == "fail"
    assert freshness["failures"] == [
        {
            "path": missing_artifact,
            "missing": True,
            "stale_references": [],
            "missing_required_artifacts": list(
                CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS
            ),
        }
    ]


def test_current_dsv4_proof_artifact_freshness_non_circular_mode_skips_manifest(
    tmp_path,
):
    for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS:
        if artifact == CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT:
            continue
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS) + "\n",
            encoding="utf-8",
        )

    freshness = validate_current_dsv4_proof_artifact_freshness(
        tmp_path,
        require_manifest_artifact=False,
    )

    assert freshness["status"] == "pass"
    assert freshness["require_manifest_artifact"] is False
    assert freshness["checked_artifacts"] == [
        artifact
        for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS
        if artifact != CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT
    ]


def test_current_proof_sweep_fails_when_dsv4_proof_artifact_freshness_fails(
    monkeypatch,
    tmp_path,
):
    freshness_failure = {
        "status": "fail",
        "artifacts": [],
        "checked_artifacts": [],
        "require_manifest_artifact": False,
        "stale_markers": ["second-local-check"],
        "required_artifacts": list(CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS),
        "failures": [
            {
                "path": CURRENT_REGRESSION_SUITE_ARTIFACT,
                "missing": False,
                "stale_references": ["second-local-check"],
                "missing_required_artifacts": [],
            }
        ],
    }

    def fail_freshness(root: Path, *, require_manifest_artifact: bool = True):
        assert root == tmp_path
        assert require_manifest_artifact is False
        return freshness_failure

    monkeypatch.setattr(
        release_regression_manifest,
        "validate_current_dsv4_proof_artifact_freshness",
        fail_freshness,
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["component_ok"]["dsv4_proof_artifact_freshness"] is False
    assert "dsv4_proof_artifact_freshness" in result["failed_components"]
    assert result["dsv4_proof_artifact_freshness"] == freshness_failure


def test_current_proof_sweep_surfaces_dsv4_proof_artifact_freshness_details(
    monkeypatch,
    tmp_path,
):
    freshness_pass = {
        "status": "pass",
        "artifacts": [],
        "checked_artifacts": [],
        "require_manifest_artifact": False,
        "stale_markers": ["second-local-check"],
        "required_artifacts": list(CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS),
        "failures": [],
    }

    def pass_freshness(root: Path, *, require_manifest_artifact: bool = True):
        assert root == tmp_path
        assert require_manifest_artifact is False
        return freshness_pass

    monkeypatch.setattr(
        release_regression_manifest,
        "validate_current_dsv4_proof_artifact_freshness",
        pass_freshness,
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["component_ok"]["dsv4_proof_artifact_freshness"] is True
    assert result["dsv4_proof_artifact_freshness"] == freshness_pass


def _write_passing_covered_live_smoke_artifacts(root: Path) -> None:
    for row_id, artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.items():
        row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS[row_id])
        result = {
            "model": row["name"],
            "row": row,
            "status": "pass",
            "failures": [],
            "command": [
                "panel/bundled-python/python/bin/python3.12",
                "-B",
                "-s",
                "-m",
                "vmlx_engine.cli",
                "serve",
                row["name"],
            ],
        }
        result["requests"] = [
            {"label": label, "validation_failures": []}
            for label in _required_live_smoke_request_labels(row_id, result)
        ]
        for request in result["requests"]:
            if request["label"] == "text_cache_repeat_1":
                request["usage"] = {"prompt_tokens_details": None}
                request["cache_summary"] = {
                    "has_cache_hit": False,
                    "cache_hit_tokens": 0,
                }
            if request["label"] == "text_cache_repeat_2":
                request["usage"] = {
                    "prompt_tokens_details": {
                        "cached_tokens": 10,
                        "cache_detail": "paged",
                    }
                }
                request["cache_summary"] = {
                    "has_cache_hit": True,
                    "cache_hit_tokens": 10,
                }
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "completed": 1,
                    "failed": 0,
                    "results": [result],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_passing_dev_ui_proof_artifacts(root: Path) -> None:
    for key, artifact in CURRENT_DEV_UI_PROOF_ARTIFACTS.items():
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("screenshot"):
            path.write_bytes(_MINIMAL_PNG)
            continue
        path.write_text(
            json.dumps(
                {
                    "repoDir": str(root.resolve()),
                    "panelDir": str((root / "panel").resolve()),
                    "appLogTail": [
                        "dev server running for the electron renderer process at:",
                        "  \u279c  Local:   http://localhost:5173/",
                        "start electron app...",
                    ],
                    "screenshots": {
                        "chatSettings": str(
                            (
                                root
                                / CURRENT_DEV_UI_PROOF_ARTIFACTS[
                                    "chat_settings_screenshot"
                                ]
                            ).resolve()
                        ),
                        "serverCacheSettings": str(
                            (
                                root
                                / CURRENT_DEV_UI_PROOF_ARTIFACTS[
                                    "server_cache_screenshot"
                                ]
                            ).resolve()
                        ),
                    },
                    "mockResponsesRequests": 2,
                    "finalAssistantContent": "Done after tools.",
                    "rawThinkTagLeak": False,
                    "persistedToolCount": 14,
                    "runToolExecutingToResultMs": 1000,
                    "eventCounts": {
                        "tool": 14,
                        "reasoningDone": 2,
                        "complete": 1,
                    },
                    "toolPhasesById": {
                        tool_id: ["calling", "executing", "result"]
                        for tool_id in (
                            "call_live_run",
                            "call_live_list",
                            "call_live_image",
                            "call_live_video",
                        )
                    },
                    "followupTailContentTypes": ["text", "image_url", "video_url"],
                    "initialPayloadKeys": {
                        "temperature": False,
                        "top_p": False,
                        "top_k": False,
                        "max_output_tokens": False,
                    },
                    "chatSettingsUi": {
                        "visible": True,
                        "labels": [
                            "Enable Built-in Coding Tools",
                            "Working Directory",
                            "Shell",
                            "Search",
                            "Utilities",
                            "Hide Tool Status",
                        ],
                        "checked": {
                            "builtinToolsEnabled": True,
                            "shellEnabled": True,
                            "fileToolsEnabled": False,
                            "gitEnabled": False,
                            "hideToolStatus": True,
                        },
                    },
                    "serverCacheUi": {
                        "visible": True,
                        "labels": [
                            "Enable Prefix Cache",
                            "Use Paged KV Cache",
                            "Block Disk Cache (L2)",
                            "Enable Disk Cache",
                            "Stored Cache Quantization",
                        ],
                        "afterBlockDiskToggle": {
                            "enablePrefixCache": True,
                            "usePagedCache": True,
                            "enableBlockDiskCache": True,
                            "enableDiskCache": False,
                        },
                        "afterDiskToggle": {
                            "enablePrefixCache": True,
                            "usePagedCache": False,
                            "enableBlockDiskCache": False,
                            "enableDiskCache": True,
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_passing_real_ui_live_model_proof_artifacts(root: Path) -> None:
    def native_cache_health(family: str | None = None) -> dict[str, object]:
        if family == "lfm25":
            return {
                "family": "lfm2_moe",
                "schema": "hybrid_ssm_v1",
                "cache_type": "hybrid_ssm_typed",
                "components": [
                    "attention_kv",
                    "ssm_companion_state",
                    "async_rederive",
                ],
                "generic_turboquant_kv": {
                    "enabled": False,
                    "reason": "hybrid_ssm_state",
                },
                "live_attention_tq_kv": {
                    "enabled": False,
                    "mode": None,
                    "applies_to": "attention_kv_layers_only",
                    "ssm_policy": "native_full_precision_companion_state",
                },
                "attention_kv_storage_quantization": {
                    "enabled": True,
                    "mode": "storage_boundary",
                    "bits": 4,
                    "group_size": 64,
                    "applies_to": "attention_kv_layers_only",
                    "ssm_policy": "native_companion_state",
                    "rederive": "async_clean_prefill_on_miss_or_warm_pass",
                },
                "prefix": True,
                "paged": True,
                "block_disk_l2": True,
                "ssm_entries": 5,
                "kv_layer_indices": [2, 6, 10, 14, 18, 21],
            }
        if family == "step37":
            return {
                "family": "mixed_attention",
                "schema": "mixed_swa_kv_v1",
                "cache_type": "mixed_swa_kv",
                "components": [
                    "full_attention_kv",
                    "sliding_window_kv",
                    "rotating_window_metadata",
                ],
                "generic_turboquant_kv": {
                    "enabled": False,
                    "reason": "not_active",
                },
                "storage_quantization": {
                    "enabled": True,
                    "mode": "storage_boundary",
                    "bits": 4,
                    "group_size": 64,
                    "applies_to": "full_attention_kv_only",
                    "metadata_policy": "preserve_rotating_window_metadata",
                },
                "prefix": True,
                "paged": True,
                "block_disk_l2": True,
            }
        return {
            "family": "plain_kv",
            "schema": "plain_kv_v1",
            "cache_type": "paged_kv",
            "components": ["attention_kv"],
            "generic_turboquant_kv": {"enabled": True, "reason": "plain_attention"},
            "prefix": True,
            "paged": True,
            "block_disk_l2": True,
        }

    def cache_endpoint_stats(family: str | None = None) -> dict[str, object]:
        stats = {
            "before": {
                "scheduler_cache": {"hits": 0, "misses": 0},
            },
            "after": {
                "scheduler_cache": {"hits": 1, "misses": 1},
                "block_disk_cache": {"disk_hits": 1, "disk_writes": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        }
        if family == "lfm25":
            stats["after"]["cache_totals"].update(
                {
                    "l2_ssm_tokens_on_disk": 64,
                    "ssm_tokens_on_disk": 64,
                    "l2_tokens_on_disk": 76,
                    "l2_tokens_on_disk_store_sum": 76,
                }
            )
            stats["after"]["ssm_companion"] = {
                "entries": 2,
                "disk_enabled": True,
                "disk": {
                    "enabled": True,
                    "entries": 2,
                    "total_tokens_on_disk": 64,
                    "total_cached_tokens": 64,
                    "stores": 2,
                },
            }
        return stats

    def add_extensive_tool_churn(proof: dict[str, object]) -> None:
        proof["eventCounts"] = {"complete": 2, "tool": 24}
        proof["persistedToolCount"] = 24
        proof["persistedToolsByMessage"] = [
            [
                *({"phase": "generating", "toolName": ""} for _ in range(9)),
                {
                    "phase": "calling",
                    "toolName": "run_command",
                    "detail": (
                        '{"command": "printf %s REAL_UI_LIVE_TOOL_ONE > '
                        'real_ui_tool_probe_1.txt && cat real_ui_tool_probe_1.txt"}'
                    ),
                },
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                        "real_ui_tool_probe_1.txt && cat real_ui_tool_probe_1.txt\n\n"
                        "REAL_UI_LIVE_TOOL_ONE"
                    ),
                },
                *({"phase": "result", "toolName": "run_command"} for _ in range(7)),
            ],
            [
                *({"phase": "generating", "toolName": ""} for _ in range(9)),
                {
                    "phase": "calling",
                    "toolName": "run_command",
                    "detail": (
                        '{"command": "printf %s REAL_UI_LIVE_TOOL_TWO > '
                        'real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt"}'
                    ),
                },
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
                *({"phase": "result", "toolName": "run_command"} for _ in range(7)),
            ],
        ]

    def add_responses_delta_streaming(proof: dict[str, object]) -> None:
        event_counts = proof.setdefault("eventCounts", {})
        event_counts["stream"] = max(int(event_counts.get("stream", 0)), 8)
        proof["streamTrace"] = [
            {
                "messageId": "synthetic-delta-one",
                "count": 4,
                "firstFullContent": "RE",
                "lastFullContent": "READY streamed progressively.",
            },
            {
                "messageId": "synthetic-delta-two",
                "count": 4,
                "firstFullContent": "OK",
                "lastFullContent": "OK second streamed turn.",
            },
        ]

    def add_generation_defaults_applied(proof: dict[str, object]) -> None:
        chat_overrides = proof.setdefault("chatOverrides", {})
        for sampler_field in ("temperature", "topP", "topK", "minP", "repeatPenalty"):
            chat_overrides.setdefault(sampler_field, None)
        request_contract = proof.get("requestContract")
        if isinstance(request_contract, dict):
            chat_overrides.setdefault("maxTokens", request_contract["requestMaxTokens"])
        route = (
            "/v1/responses"
            if proof.get("rendererWireApi") == "responses"
            else "/v1/chat/completions"
        )
        proof["serverLogTail"] = [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            f"route={route} model={proof['modelPath']} "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': "
            f"{chat_overrides.get('maxTokens', 512)}}}"
        ]

    for row in CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.values():
        screenshot = root / row["chat_screenshot"]
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(_MINIMAL_PNG)
        proof_path = root / row["proof"]
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        proof = {
                    "repoDir": str(root.resolve()),
                    "panelDir": str((root / "panel").resolve()),
                    "script": "panel/scripts/live-real-ui-model-proof.mjs",
                    "modelPath": row["model_path"],
                    "modelName": row["model_name"],
                    "server": {
                        "baseUrl": "http://127.0.0.1:8899",
                        "health": {
                            "status": "healthy",
                            "model_loaded": True,
                            "native_cache": native_cache_health(row.get("family")),
                            "kv_cache_quantization": {
                                "enabled": True,
                                "bits": 4,
                                "group_size": 64,
                            },
                            "scheduler": {
                                "last_cache_execution": {
                                    "cache_detail": (
                                        "paged+ssm"
                                        if row.get("family") == "lfm25"
                                        else "paged+mixed_swa"
                                        if row.get("family") == "step37"
                                        else "paged"
                                    ),
                                    "cached_tokens": 12,
                                }
                            },
                        },
                        "models": {"data": [{"id": row["model_name"]}]},
                    },
                    "appLogTail": [
                        "dev server running for the electron renderer process at:",
                        "  \u279c  Local:   http://localhost:5173/",
                        "start electron app...",
                    ]
                    + (
                        [
                            "[CHAT] Response complete: 134 tokens in 1.1s "
                            "(356.4 t/s, live=228.3 t/s, TTFT: 0.07s, "
                            "pp: 725 tokens (618 cached), 10984.8 pp/s, usage=server)",
                            "[CHAT] Response complete: 136 tokens in 1.1s "
                            "(353.2 t/s, live=228.3 t/s, TTFT: 0.07s, "
                            "pp: 1024 tokens (917 cached), 14027.4 pp/s, usage=server)",
                        ]
                        if row.get("family") == "lfm25"
                        else [
                            "[CHAT] Response complete: 78 tokens in 11.1s "
                            "(18.9 t/s, live=50.2 t/s, TTFT: 0.43s, "
                            "pp: 2529 tokens (2441 cached), 5813.8 pp/s, usage=server)",
                            "[CHAT] Response complete: 86 tokens in 7.7s "
                            "(19.5 t/s, live=49.9 t/s, TTFT: 0.44s, "
                            "pp: 2700 tokens (2612 cached), 6178.5 pp/s, usage=server)",
                        ]
                        if row.get("family") == "step37"
                        else []
                    ),
                    "chat": {
                        "turns": [
                            {"role": "user", "content": "Say READY in English."},
                            {"role": "assistant", "content": "READY"},
                            {"role": "user", "content": "Repeat READY once."},
                            {"role": "assistant", "content": "READY"},
                        ],
                        "finalVisibleText": "READY",
                        "rawParserTagLeak": False,
                        "cjkLeakCount": 0,
                        "koreanLeakCount": 0,
                        "reasoningText": "",
                        "reasoningRawParserTagLeak": False,
                        "reasoningCjkLeakCount": 0,
                        "reasoningKoreanLeakCount": 0,
                        "reasoningNumericRunCount": 0,
                    },
                    "cache": cache_endpoint_stats(row.get("family")),
                    "requestContract": {
                        "promptOne": "Say READY in English.",
                        "promptTwo": "Repeat READY once.",
                        "requestMaxTokens": 512,
                        "maxToolIterations": 8,
                        "toolResultMaxChars": 12000,
                        "wireApi": "chat",
                        "builtinToolsEnabled": False,
                        "enableThinking": None,
                        "checkServerCacheControls": False,
                        "checkMedia": False,
                        "checkVideo": False,
                        "expectPagedCacheLocked": False,
                        "imageExpectRegex": None,
                        "videoExpectRegex": None,
                        "cacheExpectRegex": None,
                    },
                    "screenshots": {
                        "chat": str(
                            (
                                root
                                / row["chat_screenshot"]
                            ).resolve()
                        ),
                    },
                }
        if (
            "cachecontrols" in row["proof"]
            or "filesemantic" in row["proof"]
        ):
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "zaya-text-responses-runcommand-tools-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": "$ echo REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n\n",
                    },
                ],
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && echo "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt\n\n"
                            "REAL_UI_LIVE_TOOL_ONE\n"
                        ),
                    },
                ],
            ]
        if row["proof"].endswith(
            "gemma4-responses-stricttools-max768-visible-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4, "reasoningDone": 0}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
        if row["proof"].endswith(
            "gemma4-responses-reasoningonly-max768-visible-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 0, "reasoningDone": 2}
            proof["requestedEnableThinking"] = True
            proof["requestedBuiltinTools"] = False
            proof["chatOverrides"] = {"builtinToolsEnabled": False}
            proof["persistedReasoningCount"] = 2
        if row["proof"].endswith("gemma4-cachecontrols-20260527-proof.json"):
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                ],
            }
        if row["proof"].endswith(
            (
                "zaya-text-responses-stricttools-cachecontrols-20260530-proof.json",
                "zaya-text-responses-tools-cachecontrols-disk-terminal-fix-20260601-proof.json",
                "zaya-vl-responses-stricttools-cachecontrols-20260530-proof.json",
                "step37-jang2l-devbuild-tools-image-cache-templateclean-20260531-proof.json",
                "step37-jang2l-responses-tools-l2storage-pagedlocked-after-subtype-ui-fix-20260531-proof.json",
                "lfm25-moe-a1b-jang2l-stricttools-chat-20260530-proof.json",
                "lfm25-moe-a1b-jang2l-stricttools-responses-filesemantic-20260530-proof.json",
                "lfm25-moe-a1b-jang2l-stricttools-responses-post-epipe-20260531-proof.json",
                "lfm25-moe-a1b-jang2l-devbuild-cache-parser-rerun-20260531-proof.json",
                "lfm25-moe-a1b-jang2l-responses-tools-l2storage-pagedlocked-after-content-json-ban-20260531-proof.json",
                "installed-app-lfm25-moe-a1b-jang2l-responses-tools-l2storage-cachecontrols-localonly-20260601-proof.json",
                "lfm25-jang2l-installed-responses-tools-max512-post-release-20260602-proof.json",
            )
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            if "installed-app" in row["proof"]:
                proof["uiLaunchMode"] = "installed-app"
                proof["installedAppPath"] = "/Applications/vMLX.app"
                proof["python"] = (
                    "/Applications/vMLX.app/Contents/Resources/"
                    "bundled-python/python/bin/python3"
                )
                proof["provenSurfaces"] = ["installed_app_ui"]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                ],
            }
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                            "real_ui_tool_probe_1.txt\nREAL_UI_LIVE_TOOL_ONE"
                        ),
                    },
                    {
                        "phase": "result",
                        "toolName": "write_file",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && printf %s "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && "
                            "cat real_ui_tool_probe_2.txt\n"
                            "real_ui_tool_probe_1.txt\n"
                            "real_ui_tool_probe_2.txt\n"
                            "REAL_UI_LIVE_TOOL_TWO"
                        ),
                    },
                ],
            ]
        if row["proof"].endswith(
            "step37-jang2l-devbuild-tools-image-cache-templateclean-20260531-proof.json"
        ):
            proof["media"] = {"imageVerified": True}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            add_extensive_tool_churn(proof)
        if row["proof"].endswith(
            "step37-jang2l-responses-tools-l2storage-pagedlocked-after-subtype-ui-fix-20260531-proof.json"
        ):
            add_extensive_tool_churn(proof)
        if row["proof"].endswith(
            (
                "lfm25-moe-a1b-jang2l-stricttools-chat-20260530-proof.json",
                "lfm25-moe-a1b-jang2l-stricttools-responses-post-epipe-20260531-proof.json",
                "lfm25-moe-a1b-jang2l-devbuild-cache-parser-rerun-20260531-proof.json",
                "lfm25-moe-a1b-jang2l-responses-tools-l2storage-pagedlocked-after-content-json-ban-20260531-proof.json",
                "installed-app-lfm25-moe-a1b-jang2l-responses-tools-l2storage-cachecontrols-localonly-20260601-proof.json",
                "lfm25-jang2l-installed-responses-tools-max512-post-release-20260602-proof.json",
                "lfm25-jang2l-continuation-responses-tools-cache-settings-default-max512-20260604-proof.json",
            )
        ):
            add_extensive_tool_churn(proof)
        if row["proof"].endswith(
            "step37-jang2l-responses-reasoning-max768-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 0, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = False
            proof["chatOverrides"] = {"builtinToolsEnabled": False}
            proof["persistedReasoningCount"] = 1
        if row["proof"].endswith("gemma4-cachecontrols-20260527-proof.json"):
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            (
                "nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260527-proof.json",
                "installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601-proof.json",
            )
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedReasoningCount"] = 1
            if "installed-app" in row["proof"]:
                proof["uiLaunchMode"] = "installed-app"
                proof["installedAppPath"] = "/Applications/vMLX.app"
                proof["python"] = (
                    "/Applications/vMLX.app/Contents/Resources/"
                    "bundled-python/python/bin/python3"
                )
                proof["provenSurfaces"] = ["installed_app_ui"]
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n\n",
                    },
                ],
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && printf %s "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && "
                            "cat real_ui_tool_probe_2.txt\n\n"
                            "REAL_UI_LIVE_TOOL_ONEREAL_UI_LIVE_TOOL_TWO"
                        ),
                    },
                ],
            ]
            proof["toolProbeFiles"] = {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
            }
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "ling-bailing-jangtq-responses-filesemantic-20260530-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 1, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["toolProbeFiles"] = {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE\n",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO\n",
            }
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "zaya-vl-responses-tools-cachecontrols-commandargfix-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "run_command"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedReasoningCount"] = 1
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith(
            "hy3-jangtq2-responses-tools-filesemantic-20260530-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": '$ echo "REAL_UI_LIVE_TOOL_ONE" > real_ui_tool_probe_1.txt\n\n',
                    },
                ],
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            '$ cat real_ui_tool_probe_1.txt && echo "REAL_UI_LIVE_TOOL_TWO" '
                            "> real_ui_tool_probe_2.txt\n\nREAL_UI_LIVE_TOOL_ONE\n"
                        ),
                    },
                ],
            ]
            proof["toolProbeFiles"] = {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE\n",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO\n",
            }
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
        if row["proof"].endswith("hy3-jangtq2-responses-reasoning-20260527-proof.json"):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 2, "reasoningDone": 1}
            proof["persistedReasoningCount"] = 1
        if row["proof"].endswith(
            "qwen36-mxfp4-crack-responses-tools-reasoning-image-cachecontrols-localonly-20260527-proof.json"
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 3, "tool": 4, "reasoningDone": 1}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["persistedReasoningCount"] = 1
            proof["persistedToolsByMessage"] = [
                [
                    {"phase": "result", "toolName": "run_command"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ]
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
            proof["media"] = {
                "requestedImage": True,
                "requestedVideo": False,
                "imageExpectedRegex": "\\bred\\b",
                "videoExpectedRegex": "",
                "imageSemanticVerified": True,
                "videoSemanticVerified": False,
                "imageVerified": True,
                "videoVerified": False,
                "persistedImageAttachment": True,
                "persistedVideoAttachment": False,
            }
        if row["proof"].endswith("qwen36-mxfp4-crack-video-20260527-proof.json"):
            proof["media"] = {
                "requestedImage": False,
                "requestedVideo": True,
                "imageExpectedRegex": "\\bred\\b",
                "videoExpectedRegex": "red",
                "imageSemanticVerified": False,
                "videoSemanticVerified": True,
                "imageVerified": False,
                "videoVerified": True,
                "persistedImageAttachment": False,
                "persistedVideoAttachment": True,
            }
        if (
            row.get("family") == "qwen36"
            and "20260607" in row["proof"]
        ) or (
            row.get("family") == "lfm25"
            and "lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json"
            in row["proof"]
        ):
            proof["rendererWireApi"] = "responses"
            proof["eventCounts"] = {"complete": 3, "tool": 4}
            proof["requestedBuiltinTools"] = True
            proof["chatOverrides"] = {"builtinToolsEnabled": True}
            proof["serverCacheControls"] = {
                "requested": True,
                "verified": True,
                "labels": [
                    "Enable Prefix Cache",
                    "Use Paged KV Cache",
                    "Block Disk Cache (L2)",
                    "Enable Disk Cache",
                    "Stored Cache Quantization",
                ],
            }
            proof["persistedToolsByMessage"] = [
                [
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": (
                            "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                            "real_ui_tool_probe_1.txt\nREAL_UI_LIVE_TOOL_ONE"
                        ),
                    },
                    {
                        "phase": "result",
                        "toolName": "write_file",
                        "detail": (
                            "$ cat real_ui_tool_probe_1.txt && printf %s "
                            "REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt && "
                            "cat real_ui_tool_probe_2.txt\n"
                            "real_ui_tool_probe_1.txt\n"
                            "real_ui_tool_probe_2.txt\n"
                            "REAL_UI_LIVE_TOOL_TWO"
                        ),
                    },
                ],
            ]
            if row.get("family") == "lfm25":
                add_extensive_tool_churn(proof)
            if "reasoning" in row["proof"]:
                proof["eventCounts"]["reasoningDone"] = 1
                proof["persistedReasoningCount"] = 1
            if "image" in row["proof"]:
                proof["media"] = {
                    "requestedImage": True,
                    "requestedVideo": False,
                    "imageExpectedRegex": "\\bred\\b",
                    "videoExpectedRegex": "",
                    "imageSemanticVerified": True,
                    "videoSemanticVerified": False,
                    "imageVerified": True,
                    "videoVerified": False,
                    "persistedImageAttachment": True,
                    "persistedVideoAttachment": False,
                }
            if "video" in row["proof"]:
                proof["media"] = {
                    "requestedImage": False,
                    "requestedVideo": True,
                    "imageExpectedRegex": "",
                    "videoExpectedRegex": "red",
                    "imageSemanticVerified": False,
                    "videoSemanticVerified": True,
                    "imageVerified": False,
                    "videoVerified": True,
                    "persistedImageAttachment": False,
                    "persistedVideoAttachment": True,
                }
        if row.get("family") == "zaya_vl":
            proof["media"] = {
                "requestedImage": True,
                "requestedVideo": False,
                "imageExpectedRegex": "\\bred\\b",
                "videoExpectedRegex": "",
                "imageSemanticVerified": True,
                "videoSemanticVerified": False,
                "imageVerified": True,
                "videoVerified": False,
                "persistedImageAttachment": True,
                "persistedVideoAttachment": False,
            }
            proof["server"]["health"]["scheduler"] = {
                "batch_generator": {
                    "num_images_processed": 1,
                    "vision_encoding_time": 0.01,
                }
            }
            proof["chat"]["turns"].extend(
                [
                    {
                        "role": "user",
                        "content": json.dumps(
                            [
                                {"type": "text", "text": "What color?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                    },
                                },
                            ]
                        ),
                    },
                    {"role": "assistant", "content": "Red"},
                ]
            )
            proof["chat"]["finalVisibleText"] = "Red"
        event_counts = proof.get("eventCounts")
        if isinstance(event_counts, dict) and (event_counts.get("tool") or 0) >= 3:
            add_extensive_tool_churn(proof)
        request_contract = proof["requestContract"]
        if "rendererWireApi" in proof:
            request_contract["wireApi"] = proof["rendererWireApi"]
        if "requestedBuiltinTools" in proof:
            request_contract["builtinToolsEnabled"] = proof["requestedBuiltinTools"]
        if "requestedEnableThinking" in proof:
            request_contract["enableThinking"] = proof["requestedEnableThinking"]
        chat_overrides = proof.get("chatOverrides")
        if isinstance(chat_overrides, dict):
            if "maxTokens" in chat_overrides:
                request_contract["requestMaxTokens"] = chat_overrides["maxTokens"]
            if "maxToolIterations" in chat_overrides:
                request_contract["maxToolIterations"] = chat_overrides[
                    "maxToolIterations"
                ]
            if "toolResultMaxChars" in chat_overrides:
                request_contract["toolResultMaxChars"] = chat_overrides[
                    "toolResultMaxChars"
                ]
        if proof.get("rendererWireApi") == "responses":
            add_responses_delta_streaming(proof)
        add_generation_defaults_applied(proof)
        proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")
    _write_expected_issue175_179_release_boundary_audit(root)
    _write_expected_issue181_183_runtime_audit(root)
    _write_expected_public_app_issue_audit(root)
    _write_expected_issue179_minimax_k_root_cause_audit(root)
    _write_expected_installed_app_runtime_parity_audit(root)
    _write_expected_staged_app_runtime_parity_audit(root)
    _write_expected_issue175_177_installed_runtime_audit(root)
    _write_expected_issue175_177_live_runtime_audit(root)


def _write_expected_issue175_179_release_boundary_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "issues": {
                    "175": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_memory_clear_runtime_live_stress_proven"
                        ),
                        "checks": {
                            "helper_exists": True,
                            "prefers_metal_clear_cache": True,
                            "falls_back_to_top_level_clear_cache": True,
                            "no_raw_removed_api_in_runtime_paths": True,
                            "installed_app_memory_clear_runtime_proven": True,
                            "installed_app_live_memory_stress_proven": True,
                        },
                    },
                    "176": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_promoted_block_cleanup_live_pressure_proven"
                        ),
                        "checks": {
                            "marks_promoted_disk_blocks": True,
                            "drops_disk_mirror_after_reconstruct": True,
                            "regression_test_present": True,
                            "l2_readable_write_through_regression_test_present": True,
                            "installed_app_promoted_block_cleanup_proven": True,
                            "installed_app_live_memory_pressure_proven": True,
                        },
                    },
                    "177": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_cache_selection_live_ttft_proven"
                        ),
                        "checks": {
                            "selection_telemetry_in_scheduler": True,
                            "selection_telemetry_in_health": True,
                            "cold_paged_reason_present": True,
                            "focused_regression_test_present": True,
                            "worker_timing_regression_test_present": True,
                            "installed_app_cache_selection_telemetry_proven": True,
                            "installed_live_ttft_and_cold_paged_tq_proven": True,
                        },
                    },
                    "178": {
                        "focused_source_slice": "pass",
                        "release_clearance": "packaged_app_lora_surface_proven",
                        "checks": {
                            "serve_cli_lora_paths": True,
                            "serve_cli_lora_scales": True,
                            "image_load_lora_signature_guard": True,
                            "focused_regression_test_present": True,
                            "empty_lora_lists_noop_regression_test_present": True,
                            "lora_scale_without_path_still_rejected": True,
                            "text_lora_flags_rejected": True,
                            "installed_app_lora_surface_proven": True,
                        },
                    },
                    "179": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_app_reporter_prompt_and_cancel_boundary_proven"
                        ),
                    },
                },
                "open_release_rows": [
                    "Real Electron UI cross-family live model matrix is release-cleared",
                    "DSV4 long-output/code/file-generation quality is release-cleared",
                ],
                "required_live_proof_axes": [
                    "real_app_parameters",
                    "multi_turn_chat",
                    "openai_responses_streaming",
                    "tool_parser_detection_and_execution",
                    "reasoning_parser_display_boundary",
                    "prefix_cache_reuse",
                    "paged_cache_reuse",
                    "l2_disk_cache_reconstruct",
                    "turboquant_encode_decode",
                    "hybrid_ssm_cca_swa_hsa_csa_attention",
                    "metal_matmul_runtime_path",
                    "jang_and_jangtq_metadata",
                    "vl_image_video_paths",
                    "wrong_language_and_parser_tag_leak_check",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_expected_issue181_183_runtime_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "issues": {
                    "181": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_mpp_auto_policy_guarded"
                        ),
                        "checks": {
                            "mpp_auto_policy_function_exists": True,
                            "mpp_auto_disabled_for_mxtq": True,
                            "jangtq_repo_id_disables_auto": True,
                            "explicit_mpp_on_still_allowed": True,
                            "server_health_reports_mpp_status": True,
                            "installed_app_mpp_auto_policy_disables_mxtq": True,
                        },
                    },
                    "182": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_qwen_vl_patch_embed_layout_guarded"
                        ),
                        "checks": {
                            "normal_vlm_patch_embed_transpose": True,
                            "native_mtp_patch_embed_transpose": True,
                            "focused_shape_regression_test_present": True,
                            "native_mtp_shape_regression_test_present": True,
                            "bundled_hash_gate_covers_runtime": True,
                            "installed_app_qwen_vl_patch_embed_layout": True,
                        },
                    },
                    "183": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "source_and_packaged_minicpm_v46_load_guarded"
                        ),
                        "checks": {
                            "minicpm_v46_registry_remap": True,
                            "minicpm_v46_prompt_config_remap": True,
                            "bundled_import_gate_covers_runtime": True,
                            "installed_app_minicpm_v46_runtime_remap": True,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_release_regression_manifest_accepts_issue181_183_runtime_audit(tmp_path):
    _write_expected_issue181_183_runtime_audit(tmp_path)

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert result["status"] == "pass"
    assert result["failures"] == []


def test_release_regression_manifest_rejects_issue183_missing_installed_runtime_probe(
    tmp_path,
):
    _write_expected_issue181_183_runtime_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["183"]["checks"].pop(
        "installed_app_minicpm_v46_runtime_remap"
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert "missing_issue_check:183:installed_app_minicpm_v46_runtime_remap" in result[
        "failures"
    ]


def test_release_regression_manifest_rejects_issue181_missing_installed_mpp_probe(
    tmp_path,
):
    _write_expected_issue181_183_runtime_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["181"]["checks"].pop(
        "installed_app_mpp_auto_policy_disables_mxtq"
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert "missing_issue_check:181:installed_app_mpp_auto_policy_disables_mxtq" in result[
        "failures"
    ]


def test_release_regression_manifest_rejects_issue181_missing_explicit_mpp_policy(
    tmp_path,
):
    _write_expected_issue181_183_runtime_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["181"]["checks"].pop("explicit_mpp_on_still_allowed")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert "missing_issue_check:181:explicit_mpp_on_still_allowed" in result[
        "failures"
    ]


def test_release_regression_manifest_rejects_issue182_missing_installed_qwen_vl_probe(
    tmp_path,
):
    _write_expected_issue181_183_runtime_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["182"]["checks"].pop(
        "installed_app_qwen_vl_patch_embed_layout"
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert "missing_issue_check:182:installed_app_qwen_vl_patch_embed_layout" in result[
        "failures"
    ]


def test_release_regression_manifest_rejects_issue182_missing_native_shape_marker(
    tmp_path,
):
    _write_expected_issue181_183_runtime_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["182"]["checks"].pop("native_mtp_shape_regression_test_present")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert "missing_issue_check:182:native_mtp_shape_regression_test_present" in result[
        "failures"
    ]


def test_release_regression_manifest_rejects_issue183_missing_prompt_config_marker(
    tmp_path,
):
    _write_expected_issue181_183_runtime_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["183"]["checks"].pop("minicpm_v46_prompt_config_remap")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue181_183_runtime_audit(tmp_path)

    assert "missing_issue_check:183:minicpm_v46_prompt_config_remap" in result[
        "failures"
    ]


def _write_expected_public_app_issue_audit(root: Path) -> None:
    path = root / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "issues": {
                    "165": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_dsv4_dsml_tool_call_arguments_guard"
                        ),
                        "checks": {
                            "tool_call_contract_passes": True,
                            "dsml_issue_165_regression_present": True,
                            "installed_app_dsml_parser_hash_guarded": True,
                        },
                    },
                    "166": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_gemma4_assistant_mlx_vlm_alias_guard"
                        ),
                        "checks": {
                            "source_gemma4_assistant_alias": True,
                            "bundled_verify_gemma4_assistant_alias": True,
                            "installed_app_gemma4_assistant_alias": True,
                        },
                    },
                    "169": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_supported_runtime_flavor_and_dual_staged_dmgs_guarded_packaging_still_gated"
                        ),
                        "checks": {
                            "dual_public_dmg_flavors": True,
                            "installed_app_supported_runtime_flavor": True,
                            "staged_sequoia_app_compat_runtime_flavor": True,
                            "staged_tahoe_app_native_runtime_flavor": True,
                        },
                    },
                    "111": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_mistral_small4_vlm_wrapper_detection_guard"
                        ),
                        "checks": {
                            "mistral_small4_wrapper_stays_mllm": True,
                            "mistral_small4_parser_metadata_preserved": True,
                            "installed_app_mllm_hash_guarded": True,
                        },
                    },
                    "115": {
                        "repo": "jjang-ai/mlxstudio",
                        "title": "Performance regression in v1.5.32 compared to v1.3.53",
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_current_installed_app_gemma_qwen_speed_gate"
                        ),
                        "checks": {
                            "gemma4_installed_speed_risk_tracked": True,
                            "gemma4_current_installed_ui_speed_gate_passes": True,
                            "gemma4_cold_wall_includes_ttft_tracked": True,
                            "qwen35_installed_app_speed_gate_passes": True,
                            "qwen36_speed_review_tracked": True,
                        },
                    },
                    "116": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_thinking_off_ui_api_request_guard"
                        ),
                        "checks": {
                            "reasoning_template_contract_passes": True,
                            "explicit_thinking_off_request_wired": True,
                            "panel_thinking_off_control_present": True,
                            "packaged_renderer_thinking_controls_present": True,
                        },
                    },
                    "117": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary"
                        ),
                        "checks": {"issue179_root_cause_audit_passes": True},
                    },
                    "180": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "mapped_to_minimax_small_real_ui_language_numeric_guard"
                        ),
                        "checks": {
                            "minimax_small_stricttools_real_ui_indexed": True,
                            "minimax_small_numeric_garbage_guarded": True,
                        },
                    },
                    "118": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "installed_gui_download_endpoint_and_stale_auth_fallback_guarded"
                        ),
                        "checks": {
                            "api_retries_without_stale_auth": True,
                            "installed_app_download_fallback_guarded": True,
                        },
                    },
                    "119": {
                        "focused_source_slice": "pass",
                        "release_clearance": (
                            "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
                        ),
                        "checks": {
                            "gemma26_memory_stress_artifact_present": True,
                            "gemma26_memory_stress_native_cache_health": True,
                            "gemma26_memory_stress_mixed_swa_cache_hits": True,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_release_regression_manifest_accepts_public_app_issue_audit(tmp_path):
    _write_expected_public_app_issue_audit(tmp_path)

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert result["status"] == "pass"
    assert result["failures"] == []


def test_release_regression_manifest_accepts_public_app_issue179_open_boundary(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["focused_open"] = ["117"]
    payload["issues"]["117"]["focused_source_slice"] = "open"
    payload["issues"]["117"][
        "release_clearance"
    ] = "open_minimax_k_issue179_reporter_parity_required"
    payload["issues"]["117"]["checks"]["issue179_root_cause_audit_passes"] = False
    payload["issues"]["117"]["checks"]["issue179_root_cause_audit_open"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert result["status"] == "open"
    assert result["failures"] == []


def test_release_regression_manifest_accepts_public_issue115_and_119_open_boundaries(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["focused_open"] = ["115", "119"]
    payload["issues"]["115"]["focused_source_slice"] = "open"
    payload["issues"]["115"]["checks"][
        "gemma4_current_installed_ui_speed_gate_passes"
    ] = False
    payload["issues"]["115"]["checks"][
        "gemma4_cold_wall_includes_ttft_tracked"
    ] = False
    payload["issues"]["115"]["checks"][
        "qwen35_installed_app_speed_gate_passes"
    ] = False
    payload["issues"]["119"]["focused_source_slice"] = "open"
    payload["issues"]["119"]["checks"][
        "gemma26_memory_stress_artifact_present"
    ] = False
    payload["issues"]["119"]["checks"][
        "gemma26_memory_stress_native_cache_health"
    ] = False
    payload["issues"]["119"]["checks"][
        "gemma26_memory_stress_mixed_swa_cache_hits"
    ] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert result["status"] == "open"
    assert result["failures"] == []


def test_release_regression_manifest_rejects_public_issue118_without_installed_app_download_guard(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["118"]["checks"].pop("installed_app_download_fallback_guarded")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert "missing_issue_check:118:installed_app_download_fallback_guarded" in result[
        "failures"
    ]


def test_release_regression_manifest_rejects_public_issue180_without_numeric_guard(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["180"]["checks"].pop("minimax_small_numeric_garbage_guarded", None)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert (
        "missing_issue_check:180:minimax_small_numeric_garbage_guarded"
        in result["failures"]
    )


def test_release_regression_manifest_rejects_public_issue166_without_installed_alias_guard(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["166"]["checks"].pop(
        "installed_app_gemma4_assistant_alias", None
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert (
        "missing_issue_check:166:installed_app_gemma4_assistant_alias"
        in result["failures"]
    )


def test_release_regression_manifest_rejects_public_issue111_without_installed_mllm_hash_guard(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"].pop("111", None)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert "missing_issue:111" in result["failures"]


def test_release_regression_manifest_rejects_public_issue116_without_thinking_off_guard(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"].pop("116", None)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert "missing_issue:116" in result["failures"]


def test_release_regression_manifest_does_not_block_on_cleared_public_issue115_speed_proofs():
    sweep = validate_current_proof_sweep_artifacts(Path("."))
    blockers = {
        blocker["id"]: blocker
        for blocker in sweep["release_blocker_ledger"]["blockers"]
    }

    assert "issue115_installed_app_performance_regression_open" not in blockers


def test_release_regression_manifest_rejects_public_issue169_without_installed_runtime_flavor(
    tmp_path,
):
    _write_expected_public_app_issue_audit(tmp_path)
    path = tmp_path / CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["169"]["checks"].pop(
        "installed_app_supported_runtime_flavor"
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_public_app_issue_audit(tmp_path)

    assert (
        "missing_issue_check:169:installed_app_supported_runtime_flavor"
        in result["failures"]
    )


def test_release_regression_manifest_accepts_open_issue175_179_boundary_audit(
    tmp_path,
):
    _write_expected_issue175_179_release_boundary_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["issues"]["179"]["focused_source_slice"] = "open"
    payload["issues"]["179"]["release_clearance"] = "open_reporter_parity_required"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue175_179_release_boundary_audit(tmp_path)

    assert result["status"] == "open"
    assert result["failures"] == []
    assert "issue175_179_boundary_still_open" not in result["failures"]


def test_release_regression_manifest_rejects_issue177_without_live_ttft_tq_guard(
    tmp_path,
):
    _write_expected_issue175_179_release_boundary_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["issues"]["177"]["checks"].pop(
        "installed_live_ttft_and_cold_paged_tq_proven"
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _validate_current_issue175_179_release_boundary_audit(tmp_path)

    assert (
        "missing_issue_check:177:installed_live_ttft_and_cold_paged_tq_proven"
        in result["failures"]
    )


def _write_expected_issue179_minimax_k_root_cause_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "proven": {
                    "reporter_log_installed_app_bundled_python_seen": True,
                    "reporter_log_request_shape_and_sampling_kwargs_seen": True,
                    "reporter_log_launch_parser_cache_flags_seen": True,
                    "reporter_log_has_abort_before_visible_content": True,
                    "reporter_log_has_responses_cancel_404": True,
                    "local_real_ui_diagnostics_clean": True,
                    "local_installed_issue179_session_settings_parity": True,
                    "local_installed_bundle_has_responses_cancel_route": True,
                    "public_v1549_tahoe_dmg_has_responses_cancel_route": True,
                    "latest_public_dmg_has_responses_cancel_route": True,
                    "local_installed_responses_cancel_live_probe": True,
                    "current_source_responses_cancel_contract_proven": True,
                    "current_source_responses_cancel_inactive_404_contract_proven": True,
                    "reporter_cancel_404_after_stream_abort_order_proven": True,
                    "reporter_cancel_404_after_econnreset_same_response_id_proven": True,
                    "local_reporter_prompt_reproduction_clean": True,
                },
                "not_proven": [],
                "local_reporter_prompt_reproduction": {
                    "path": "build/current-issue179-minimax-k-responses-cancel-probe-current-source-parser-settings-parity-20260608.json",
                    "exists": True,
                    "status": "pass",
                    "request_matches_reporter": True,
                    "stream_started": True,
                    "response_id_seen": True,
                    "cancel_status": 200,
                    "cancel_trigger": "delay_elapsed",
                    "bad_text_captured": False,
                    "bad_text_counts": {
                        "raw_parser_tag": False,
                        "cjk": 0,
                        "korean": 0,
                        "numeric": 0,
                    },
                    "observed_stream_text": True,
                    "clean": True,
                },
                "local_responses_cancel_probe": {
                    "path": "build/current-issue179-minimax-k-responses-cancel-probe-current-source-parser-settings-parity-20260608.json",
                    "exists": True,
                    "status": "pass",
                    "response_id_seen": True,
                    "cancel_status": 200,
                    "probe": {
                        "abort_boundary": "controlled_cancel_after_response_id",
                        "bad_text_captured": False,
                        "cancel_route_present": True,
                    },
                },
                "reporter_parity_artifact": {
                    "path": "build/issue-179/reporter-parity-metadata-20260527.json",
                    "exists": True,
                    "status": "pass",
                    "comparison_status": "ready_for_direct_comparison",
                    "collector_missing_fields": [],
                    "required_fields": [
                        "capture_provenance",
                        "installed_server_sha256",
                        "server_has_responses_cancel_route",
                        "server_cancel_calls_engine_abort",
                        "model_manifest_sha256",
                        "model_file_hashes",
                        "chat_id",
                        "session_settings",
                        "response_id",
                        "response_active_at_cancel",
                        "raw_sse_cancel_lifecycle",
                    ],
                },
                "reporter_parity_comparison": {
                    "status": "pass",
                    "capture_provenance_is_reporter_machine": True,
                    "server_hash_matches_local_installed": True,
                    "server_route_markers_match": True,
                    "model_manifest_sha256_matches_local": True,
                    "model_file_hashes_match_local": True,
                    "chat_id_matches_reporter_log": True,
                    "response_id_matches_reporter_log": True,
                    "response_active_at_cancel_recorded": True,
                    "raw_sse_cancel_lifecycle_present": True,
                    "failures": [],
                },
                "reporter_server_hash_parity": {
                    "status": "pass",
                    "reporter_installed_server_sha256": "server-sha",
                    "source_server_sha256": "server-sha",
                    "local_installed_server_sha256": "server-sha",
                    "public_v1549_tahoe_server_sha256": "public-sha",
                    "latest_public_server_sha256": "server-sha",
                    "reporter_matches_source": True,
                    "reporter_matches_local_installed": True,
                    "reporter_matches_public_v1549_tahoe": False,
                    "reporter_matches_latest_public": True,
                    "route_markers_match": True,
                    "provenance": {
                        "status": "pass",
                        "checked_sources": [
                            "source_contract",
                            "local_installed_bundle",
                            "public_v1549_tahoe_dmg",
                            "public_release_dmg_contracts",
                            "local_installed_app_backups",
                            "sibling_python_worktrees",
                            "git_history",
                        ],
                        "direct_matches": {
                            "source": True,
                            "local_installed": True,
                            "public_v1549_tahoe": False,
                        },
                        "local_backup_matches": [],
                        "local_backup_checked_count": 0,
                        "git_history": {
                            "checked": True,
                            "match": True,
                            "commit": "abc123",
                            "error": None,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_issue179_minimax_k_live_probe_memory_preflight(root: Path) -> None:
    path = root / CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason": "insufficient_vm_stat_memory",
                "launch_allowed": False,
                "launch_decision": "do_not_launch",
                "model_path": "/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K",
                "model_size_gb": 73.96,
                "required_free_gb": 79.96,
                "min_free_gb": 79.96,
                "available_for_gate_gb": 73.54,
                "free_plus_speculative_purgeable_gb": 73.54,
                "memory_gap_gb": 6.42,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "commands": {
                    "memory": "vm_stat",
                    "top_memory_processes": "ps -axo pid,rss,command -r | head -n 11",
                    "active_heavy_processes": "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|run_runtime_memory_stress_probe|run_decode_speed_gate|live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|run_dsv4_default_cache_tool_loop_gate|run_current_regression_suite|run_issue179_responses_cancel_probe'",
                },
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 2.0, "command": "codex --yolo resume"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "preflight_captured_at": "2026-06-01T170000-0700",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_step37_vlm_runtime_audit(root: Path) -> None:
    path = root / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "step37_vlm_runtime_present_or_not_applicable",
                "release_clearance": "audit_does_not_block_release",
                "mlx_vlm_step3p7_importable": True,
                "mlx_vlm_step3p7_runtime_available": True,
                "mlx_vlm_step3p7_runtime": "available",
                "local_bridge_kind": "native_or_not_applicable",
                "live_media_proof": {
                    "pass": True,
                    "status": "pass",
                    "failed": 0,
                    "validation_failures": [],
                    "present_labels": [
                        "text_cache_repeat_1",
                        "text_cache_repeat_2",
                        "text_multiturn_recall",
                        "tool_required",
                        "tool_result_continuation",
                        "structured_json_exact",
                        "exact_code_whitespace",
                        "vl_blue_image",
                        "text_no_media_after_image",
                        "vl_blue_image_repeat",
                        "vl_red_image_changed",
                        "vl_blue_video",
                        "text_no_media_after_video",
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_current_proof_sweep_rejects_issue179_missing_live_cancel_probe(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"].pop("local_installed_responses_cancel_live_probe")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:local_installed_responses_cancel_live_probe"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_public_dmg_route_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"].pop("public_v1549_tahoe_dmg_has_responses_cancel_route")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:public_v1549_tahoe_dmg_has_responses_cancel_route"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_econnreset_same_id_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_cancel_404_after_econnreset_same_response_id_proven"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_cancel_404_after_econnreset_same_response_id_proven"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_installed_app_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_log_installed_app_bundled_python_seen"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_log_installed_app_bundled_python_seen"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_settings_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_log_request_shape_and_sampling_kwargs_seen"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_log_request_shape_and_sampling_kwargs_seen"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_launch_config_proof(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["reporter_log_launch_parser_cache_flags_seen"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:reporter_log_launch_parser_cache_flags_seen"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_local_installed_session_parity(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proven"]["local_installed_issue179_session_settings_parity"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert (
        "missing_proven:local_installed_issue179_session_settings_parity"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_issue179_missing_reporter_parity_artifact_contract(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reporter_parity_artifact")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_parity_artifact_contract" in result["failures"]


def test_current_proof_sweep_rejects_issue179_missing_local_responses_cancel_probe(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("local_responses_cancel_probe")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_local_responses_cancel_probe" in result["failures"]


def test_current_proof_sweep_rejects_issue179_failed_local_responses_cancel_probe(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["local_responses_cancel_probe"]["cancel_status"] = 404
    payload["local_responses_cancel_probe"]["probe"]["bad_text_captured"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "local_responses_cancel_probe_cancel_not_200" in result["failures"]
    assert "local_responses_cancel_bad_text_captured" in result["failures"]


def test_current_proof_sweep_rejects_issue179_missing_reporter_parity_comparison(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reporter_parity_comparison")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_parity_comparison" in result["failures"]


def test_current_proof_sweep_rejects_issue179_missing_reporter_server_hash_parity(
    tmp_path,
):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reporter_server_hash_parity")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_server_hash_parity" in result["failures"]


def test_current_proof_sweep_rejects_issue179_missing_reporter_server_hash_provenance(
    tmp_path,
):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reporter_server_hash_parity"].pop("provenance")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_server_hash_provenance" in result["failures"]


def test_current_proof_sweep_rejects_issue179_narrow_public_dmg_hash_coverage(
    tmp_path,
):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["not_proven"] = [
        "reporter installed app bundle hash matches public/local server.py route proof"
    ]
    payload["reporter_server_hash_parity"]["status"] = "open"
    payload["reporter_server_hash_parity"][
        "failure"
    ] = "reporter_installed_server_hash_drift"
    payload["reporter_server_hash_parity"]["provenance"] = {
        "status": "open",
        "checked_sources": [
            "source_contract",
            "local_installed_bundle",
            "public_v1549_tahoe_dmg",
            "public_release_dmg_contracts",
            "local_installed_app_backups",
            "sibling_python_worktrees",
            "git_history",
        ],
        "public_release_checked_count": 4,
        "failure": "reporter_server_hash_provenance_unknown",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "insufficient_public_release_dmg_contract_coverage" in result["failures"]


def test_current_proof_sweep_rejects_issue179_missing_sibling_worktree_hash_coverage(
    tmp_path,
):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    checked_sources = payload["reporter_server_hash_parity"]["provenance"][
        "checked_sources"
    ]
    payload["reporter_server_hash_parity"]["provenance"]["checked_sources"] = [
        source for source in checked_sources if source != "sibling_python_worktrees"
    ]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_root_cause_audit"
    ]

    assert "missing_reporter_server_hash_provenance_sources" in result["failures"]


def test_current_proof_sweep_surfaces_issue179_open_not_proven_details(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    not_proven = (
        "reporter installed app bundle hash matches public/local server.py route proof"
    )
    payload["status"] = "open"
    payload["not_proven"] = [not_proven]
    payload["release_boundary"] = (
        "#179 remains open until reporter bundle hash provenance is proven."
    )
    payload["reporter_server_hash_parity"]["status"] = "open"
    payload["reporter_server_hash_parity"][
        "failure"
    ] = "reporter_installed_server_hash_drift"
    payload["reporter_server_hash_parity"]["provenance"] = {
        "status": "open",
        "checked_sources": [
            "source_contract",
            "local_installed_bundle",
            "public_v1549_tahoe_dmg",
            "public_release_dmg_contracts",
            "local_installed_app_backups",
            "sibling_python_worktrees",
            "git_history",
        ],
        "public_release_checked_count": 15,
        "failure": "reporter_server_hash_provenance_unknown",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    sweep = validate_current_proof_sweep_artifacts(tmp_path)
    result = sweep["issue179_minimax_k_root_cause_audit"]

    assert result["status"] == "open"
    assert result["not_proven"] == [not_proven]
    assert result["release_boundary"] == (
        "#179 remains open until reporter bundle hash provenance is proven."
    )
    assert result["failures"] == []
    blockers = {gap["id"]: gap for gap in sweep["release_blocker_ledger"]["blockers"]}
    assert blockers["issue179_minimax_k_root_cause_audit"]["details"][
        "not_proven"
    ] == [not_proven]
    assert blockers["issue179_minimax_k_root_cause_audit"]["details"][
        "release_boundary"
    ] == "#179 remains open until reporter bundle hash provenance is proven."
    assert blockers["issue179_minimax_k_root_cause_audit"]["details"][
        "reporter_server_hash_parity"
    ]["failure"] == "reporter_installed_server_hash_drift"
    assert "reporter installed app bundle hash provenance" in blockers[
        "issue179_minimax_k_root_cause_audit"
    ]["next_proof"]
    assert "broad cache/model probes" in blockers[
        "issue179_minimax_k_root_cause_audit"
    ]["next_proof"]


def test_current_proof_sweep_requires_issue179_memory_preflight_when_open(tmp_path):
    _write_expected_issue179_minimax_k_root_cause_audit(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["not_proven"] = [
        "reporter installed app bundle hash matches public/local server.py route proof"
    ]
    payload["reporter_server_hash_parity"]["status"] = "open"
    payload["reporter_server_hash_parity"][
        "failure"
    ] = "reporter_installed_server_hash_drift"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["issue179_minimax_k_live_probe_memory_preflight"]["status"] == "missing"
    assert (
        result["issue179_minimax_k_live_probe_memory_preflight"]["artifact"]
        == CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    )
    assert result["component_ok"]["issue179_minimax_k_live_probe_memory_preflight"] is False
    assert result["status"] == "fail"


def test_current_proof_sweep_accepts_issue179_memory_preflight_ready(tmp_path):
    _write_passing_issue179_minimax_k_live_probe_memory_preflight(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "ready_to_launch",
            "reason": "memory_preflight_floor_met",
            "available_for_gate_gb": 83.83,
            "free_plus_speculative_purgeable_gb": 83.83,
            "memory_gap_gb": 0.0,
            "launch_blockers": [],
            "launch_decision": "launch_allowed",
            "launch_allowed": True,
        }
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_live_probe_memory_preflight"
    ]

    assert result == {
        "artifact": CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT,
        "status": "pass",
        "failures": [],
        "reason": "memory_preflight_floor_met",
        "model_path": "/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K",
        "model_size_gb": 73.96,
        "required_free_gb": 79.96,
        "min_free_gb": 79.96,
        "available_for_gate_gb": 83.83,
        "free_plus_speculative_purgeable_gb": 83.83,
        "memory_gap_gb": 0.0,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        "commands": {
            "memory": "vm_stat",
            "top_memory_processes": "ps -axo pid,rss,command -r | head -n 11",
            "active_heavy_processes": "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|run_runtime_memory_stress_probe|run_decode_speed_gate|live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|run_dsv4_default_cache_tool_loop_gate|run_current_regression_suite|run_issue179_responses_cancel_probe'",
        },
        "top_memory_processes": [
            {"pid": 1001, "rss_gb": 2.0, "command": "codex --yolo resume"}
        ],
        "active_heavy_processes": [],
        "active_heavy_process_count": 0,
        "launch_blockers": [],
        "preflight_captured_at": "2026-06-01T170000-0700",
        "launch_decision": "launch_allowed",
        "launch_allowed": True,
    }


def test_current_proof_sweep_rejects_issue179_memory_preflight_without_process_context(
    tmp_path,
):
    _write_passing_issue179_minimax_k_live_probe_memory_preflight(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("commands", None)
    payload.pop("top_memory_processes", None)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_live_probe_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert result["failures"] == [
        "top_memory_processes_missing",
        "commands_missing",
    ]


def test_current_proof_sweep_rejects_issue179_memory_preflight_without_capture_time(
    tmp_path,
):
    _write_passing_issue179_minimax_k_live_probe_memory_preflight(tmp_path)
    path = tmp_path / CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("preflight_captured_at", None)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "issue179_minimax_k_live_probe_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert result["failures"] == ["preflight_captured_at_missing"]


def _write_expected_installed_app_runtime_parity_audit(root: Path) -> None:
    path = root / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "installed_python_exists": True,
                    "installed_versioned_python_exists": True,
                    "installed_versioned_python_runs": True,
                    "installed_bundled_python_launch_crash_reports_classified": True,
                    "installed_bundled_python_launch_crashes_not_reproduced": True,
                    "serve_help_runs": True,
                    "responses_cancel_route": True,
                    "image_lora_cli_flags": True,
                    "image_lora_load_signature": True,
                    "image_lora_server_globals": True,
                    "text_lora_flags_rejected": True,
                    "xml_function_tool_parser_cli": True,
                    "plain_attention_kv_native_cache_status": True,
                    "installed_panel_gateway_epipe_guard": True,
                    "installed_panel_gateway_epipe_aggregate_guard": True,
                    "installed_panel_chat_ipc_epipe_aggregate_guard": True,
                    "installed_panel_image_ipc_epipe_aggregate_guard": True,
                    "installed_panel_cache_ipc_epipe_aggregate_guard": True,
                    "installed_panel_performance_health_epipe_guard": True,
                    "installed_panel_session_lifecycle_epipe_guard": True,
                    "installed_panel_child_process_stdio_epipe_guard": True,
                    "installed_panel_child_process_stdio_epipe_aggregate_guard": True,
                    "installed_panel_renderer_chat_epipe_toast_normalized": True,
                    "installed_panel_responses_stream_cache_detail_metrics": True,
                    "installed_panel_gateway_guarded_proxy_forwarding": True,
                    "installed_panel_gateway_write_once_behavior_marker": True,
                    "installed_panel_gateway_response_socket_destroyed_guard": True,
                    "installed_panel_gateway_request_socket_destroyed_guard": True,
                    "installed_panel_gateway_single_model_cache_endpoint_routing": True,
                    "installed_panel_gateway_single_model_streaming_routes": True,
                    "installed_panel_max_output_context_settings_wired": True,
                    "installed_panel_model_owned_generation_defaults_wired": True,
                    "installed_panel_request_builders_keep_output_context_separate": True,
                    "installed_panel_parser_reasoning_settings_wired": True,
                    "installed_panel_ollama_streaming_json_writes_guarded": True,
                    "installed_panel_ollama_proxy_response_errors_guarded": True,
                    "installed_vmlx_user_data_no_raw_disconnect_errors": True,
                    "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors": True,
                },
                "missing_or_stale": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_expected_staged_app_runtime_parity_audit(root: Path) -> None:
    path = root / CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "installed_app": str(
                    root / "panel/release/sequoia-app/mac-arm64/vMLX.app"
                ),
                "checks": {
                    "installed_python_exists": True,
                    "installed_versioned_python_exists": True,
                    "installed_versioned_python_runs": True,
                    "installed_bundled_python_launch_crash_reports_classified": True,
                    "installed_bundled_python_launch_crashes_not_reproduced": True,
                    "serve_help_runs": True,
                    "responses_cancel_route": True,
                    "image_lora_cli_flags": True,
                    "image_lora_load_signature": True,
                    "image_lora_server_globals": True,
                    "text_lora_flags_rejected": True,
                    "xml_function_tool_parser_cli": True,
                    "plain_attention_kv_native_cache_status": True,
                    "installed_panel_gateway_epipe_guard": True,
                    "installed_panel_gateway_epipe_aggregate_guard": True,
                    "installed_panel_chat_ipc_epipe_aggregate_guard": True,
                    "installed_panel_image_ipc_epipe_aggregate_guard": True,
                    "installed_panel_cache_ipc_epipe_aggregate_guard": True,
                    "installed_panel_performance_health_epipe_guard": True,
                    "installed_panel_session_lifecycle_epipe_guard": True,
                    "installed_panel_child_process_stdio_epipe_guard": True,
                    "installed_panel_child_process_stdio_epipe_aggregate_guard": True,
                    "installed_panel_renderer_chat_epipe_toast_normalized": True,
                    "installed_panel_responses_stream_cache_detail_metrics": True,
                    "installed_panel_gateway_guarded_proxy_forwarding": True,
                    "installed_panel_gateway_write_once_behavior_marker": True,
                    "installed_panel_gateway_response_socket_destroyed_guard": True,
                    "installed_panel_gateway_request_socket_destroyed_guard": True,
                    "installed_panel_gateway_single_model_cache_endpoint_routing": True,
                    "installed_panel_gateway_single_model_streaming_routes": True,
                    "installed_panel_max_output_context_settings_wired": True,
                    "installed_panel_model_owned_generation_defaults_wired": True,
                    "installed_panel_request_builders_keep_output_context_separate": True,
                    "installed_panel_parser_reasoning_settings_wired": True,
                    "installed_panel_ollama_streaming_json_writes_guarded": True,
                    "installed_panel_ollama_proxy_response_errors_guarded": True,
                    "installed_vmlx_user_data_no_raw_disconnect_errors": True,
                    "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors": True,
                },
                "missing_or_stale": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_current_proof_sweep_requires_staged_app_runtime_parity_audit(tmp_path):
    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "staged_app_runtime_parity_audit"
    ]

    assert result == {
        "artifact": CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        "status": "missing",
        "missing": [CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT],
        "failures": [],
    }


def test_current_proof_sweep_accepts_staged_app_runtime_parity_pass(tmp_path):
    _write_expected_staged_app_runtime_parity_audit(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "staged_app_runtime_parity_audit"
    ]

    assert result == {
        "artifact": CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }


def test_current_proof_sweep_rejects_installed_app_missing_generation_settings_parity(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_model_owned_generation_defaults_wired")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_model_owned_generation_defaults_wired"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_parser_settings_parity(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_parser_reasoning_settings_wired")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_parser_reasoning_settings_wired"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_versioned_python_launch(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_versioned_python_runs")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert "missing_check:installed_versioned_python_runs" in result["failures"]


def test_current_proof_sweep_rejects_installed_app_missing_bundled_python_crash_classification(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop(
        "installed_bundled_python_launch_crash_reports_classified"
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_bundled_python_launch_crash_reports_classified"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_bundled_python_crash_repro_check(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_bundled_python_launch_crashes_not_reproduced")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_bundled_python_launch_crashes_not_reproduced"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_epipe_aggregate_guard(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_gateway_epipe_aggregate_guard")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_gateway_epipe_aggregate_guard"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_ipc_epipe_aggregate_guard(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_chat_ipc_epipe_aggregate_guard")
    payload["checks"].pop("installed_panel_image_ipc_epipe_aggregate_guard")
    payload["checks"].pop("installed_panel_cache_ipc_epipe_aggregate_guard")
    payload["checks"].pop("installed_panel_performance_health_epipe_guard")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_chat_ipc_epipe_aggregate_guard"
        in result["failures"]
    )
    assert (
        "missing_check:installed_panel_image_ipc_epipe_aggregate_guard"
        in result["failures"]
    )
    assert (
        "missing_check:installed_panel_cache_ipc_epipe_aggregate_guard"
        in result["failures"]
    )
    assert (
        "missing_check:installed_panel_performance_health_epipe_guard"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_child_stdio_epipe_guard(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_panel_child_process_stdio_epipe_guard")
    payload["checks"].pop("installed_panel_child_process_stdio_epipe_aggregate_guard")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_child_process_stdio_epipe_guard"
        in result["failures"]
    )
    assert (
        "missing_check:installed_panel_child_process_stdio_epipe_aggregate_guard"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_renderer_epipe_toast_guard(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop(
        "installed_panel_renderer_chat_epipe_toast_normalized",
        None,
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_renderer_chat_epipe_toast_normalized"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_responses_cache_detail_metrics(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop(
        "installed_panel_responses_stream_cache_detail_metrics",
        None,
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_panel_responses_stream_cache_detail_metrics"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_open_installed_app_runtime_parity(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "open"
    payload["missing_or_stale"] = [
        "installed_panel_renderer_chat_epipe_toast_normalized"
    ]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert "installed_app_parity_not_pass" in result["failures"]


def test_current_proof_sweep_rejects_installed_app_missing_user_data_epipe_scan(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_vmlx_user_data_no_raw_disconnect_errors")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_vmlx_user_data_no_raw_disconnect_errors"
        in result["failures"]
    )


def test_current_proof_sweep_rejects_installed_app_missing_diagnostic_epipe_scan(
    tmp_path,
):
    _write_expected_installed_app_runtime_parity_audit(tmp_path)
    path = tmp_path / CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"].pop("installed_vmlx_diagnostic_reports_no_raw_disconnect_errors")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "installed_app_runtime_parity_audit"
    ]

    assert (
        "missing_check:installed_vmlx_diagnostic_reports_no_raw_disconnect_errors"
        in result["failures"]
    )


def _write_expected_issue175_177_installed_runtime_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "installed_python_exists": True,
                    "memory_clear_helper_imports_from_installed_app": True,
                    "memory_clear_uses_available_mlx_api": True,
                    "runtime_paths_avoid_removed_clear_memory_cache": True,
                    "promoted_disk_blocks_drop_parent_mirror": True,
                    "l2_readable_write_through_blocks_drop_parent_mirror": True,
                    "cache_selection_telemetry_installed": True,
                    "cache_execution_timing_telemetry_installed": True,
                },
                "failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_expected_issue175_177_live_runtime_audit(root: Path) -> None:
    path = root / CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "installed_app_qwen_live_probe_passed": True,
                    "cache_hit_ttft_improved": True,
                    "l2_block_disk_hits_observed": True,
                    "paged_ssm_cache_hit_observed": True,
                    "visible_content_observed": True,
                    "metal_memory_metrics_observed": True,
                    "prefix_paged_l2_capacity_projection_valid": True,
                    "minimax_installed_live_probe_passed": True,
                    "turboquant_live_decode_observed": True,
                    "scheduler_cache_selection_observed": True,
                    "scheduler_cache_execution_timing_observed": True,
                    "paged_tq_cache_hit_observed": True,
                    "minimax_l2_block_disk_hits_observed": True,
                    "minimax_cache_hit_request_latency_improved": True,
                    "minimax_visible_content_observed": True,
                    "minimax_restart_reader_probe_passed": True,
                    "cold_paged_tq_restart_hit_observed": True,
                    "cold_paged_cache_selection_observed": True,
                    "cold_paged_stream_ttft_observed": True,
                    "restart_visible_content_observed": True,
                    "admin_sleep_lifecycle_probe_passed": True,
                    "admin_sleep_deep_unload_observed": True,
                },
                "remaining_blockers": [],
                "failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_dsv4_source_memory_preflight_artifact(root: Path) -> None:
    path = root / CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason": "insufficient_vm_stat_memory",
                "model": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "commands": {
                    "memory": "vm_stat",
                    "top_memory_processes": "ps -axo pid,rss,command -r | head -n 11",
                    "active_heavy_processes": "pgrep -af 'vmlx_engine.cli serve|mlx_lm.server|run_runtime_memory_stress_probe|run_decode_speed_gate|live-real-ui-model-proof|run_dsv4_route_mode_code_exactness|run_dsv4_default_cache_tool_loop_gate'",
                    "memory_pressure": "memory_pressure",
                },
                "available_gb": 105.0,
                "required_available_gb": 120.0,
                "required_free_gb": 120.0,
                "min_free_gb": 120.0,
                "available_for_gate_gb": 79.25,
                "memory_gap_gb": 40.75,
                "strict_vm_stat_memory_gap_gb": 40.75,
                "psutil_available_gap_gb": 15.0,
                "free_plus_speculative_purgeable_gb": 79.25,
                "memory_pressure_free_percent": 95,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "did_not_launch": True,
                "launch_decision": "do_not_launch",
                "launch_allowed": False,
                "launch_blockers": ["insufficient_memory"],
                "active_heavy_process_count": 0,
                "active_heavy_processes": [],
                "top_memory_processes": [
                    {"pid": 4321, "rss_gb": 2.0, "command": "current-codex"}
                ],
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
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_real_ui_dsv4_memory_preflight_artifact(root: Path) -> None:
    path = root / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "generated_at": "2026-06-01T17:56:03.986322+00:00",
                "model_path": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 80.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "required_available_gb": 120.0,
                "required_free_gb": 120.0,
                "min_free_gb": 120.0,
                "memory_gap_gb": 80.85,
                "available_for_gate_gb": 39.15,
                "psutil_available_gb": 104.0,
                "psutil_memory_gap_gb": 16.0,
                "memory_pressure_free_percent": 95,
                "inactive_file_cache_gb": 30.0,
                "launch_decision": "do_not_launch",
                "launch_allowed": False,
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_passing_covered_live_tool_smoke_artifacts(root: Path) -> None:
    for row_id, artifact in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.items():
        row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS[row_id])
        live_labels = (
            _required_live_smoke_request_labels(row_id, {"row": row})
            if artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.values()
            else ["text_cache_repeat_1", "text_cache_repeat_2", "text_multiturn_recall"]
        )
        result = {
            "model": row["name"],
            "row": row,
            "status": "pass",
            "failures": [],
            "command": [
                "panel/bundled-python/python/bin/python3.12",
                "-B",
                "-s",
                "-m",
                "vmlx_engine.cli",
                "serve",
                row["name"],
            ],
            "requests": [
                *[
                    {"label": label, "validation_failures": []}
                    for label in live_labels
                ],
                {
                    "label": "tool_required",
                    "content_head": "",
                    "validation_failures": [],
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "record_fact",
                                "arguments": json.dumps({"value": "blue-cat"}),
                            },
                        }
                    ],
                },
            ],
        }
        for request in result["requests"]:
            if request["label"] == "text_cache_repeat_1":
                request["usage"] = {"prompt_tokens_details": None}
                request["cache_summary"] = {
                    "has_cache_hit": False,
                    "cache_hit_tokens": 0,
                }
            if request["label"] == "text_cache_repeat_2":
                request["usage"] = {
                    "prompt_tokens_details": {
                        "cached_tokens": 10,
                        "cache_detail": "paged",
                    }
                }
                request["cache_summary"] = {
                    "has_cache_hit": True,
                    "cache_hit_tokens": 10,
                }
        if row.get("supports_thinking"):
            result["requests"].insert(
                3,
                {"label": "reasoning_on", "validation_failures": []},
            )
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "completed": 1,
                    "failed": 0,
                    "results": [result],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_expected_diagnostic_live_smoke_artifacts(root: Path) -> None:
    for row_id, expectation in CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS.items():
        artifact = expectation["artifact"]
        failures = []
        for label, reason_or_reasons in expectation["expected_label_reasons"].items():
            reasons = (
                reason_or_reasons
                if isinstance(reason_or_reasons, list)
                else [reason_or_reasons]
            )
            failures.extend(
                {"label": label, "reason": reason}
                for reason in reasons
            )
        path = root / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": expectation["expected_status"],
                    "completed": 1,
                    "failed": 1,
                    "results": [
                        {
                            "model": row_id,
                            "status": expectation["expected_status"],
                            "failures": failures,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _write_passing_mimo_v2_root_cause_artifacts(root: Path) -> None:
    structural_path = root / CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT
    structural_path.parent.mkdir(parents=True, exist_ok=True)
    structural_path.write_text(
        json.dumps({"status": "pass", "files": 173}) + "\n",
        encoding="utf-8",
    )
    text_cache_path = root / CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT
    text_cache_path.parent.mkdir(parents=True, exist_ok=True)
    text_cache_path.write_text(
        json.dumps(
            {
                "requests": [
                    {"content": "cache ok", "usage": {}},
                    {
                        "content": "cache ok",
                        "usage": {
                            "prompt_tokens_details": {
                                "cached_tokens": 28,
                                "cache_detail": "paged",
                            }
                        },
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    switchglu_path = root / CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT
    switchglu_path.parent.mkdir(parents=True, exist_ok=True)
    switchglu_path.write_text(
        json.dumps({"max_abs_diff": 0.0007, "mean_abs_diff": 0.00009}) + "\n",
        encoding="utf-8",
    )
    length_sweep_path = root / CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT
    length_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    length_sweep_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "cases": [
                    {"prompt_tokens": 60, "status": "pass", "output": "length ok"},
                    {"prompt_tokens": 180, "status": "pass", "output": "length ok"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tool_path = root / CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "runtime_observations": [
                    {"http_status": 200, "tool_calls": [{"id": "call_1"}]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_audit_path = root / CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    current_audit_path.parent.mkdir(parents=True, exist_ok=True)
    current_audit_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "local_release_clearance": True,
                "component_ok": {
                    "manifest_integrity": True,
                    "stale_local_state_absent": True,
                    "structural_verify": True,
                    "text_cache_narrow": True,
                    "switchglu_selected_expert_parity": True,
                    "cache_vs_nocache_next_token": True,
                    "long_prompt_coherence": True,
                    "tool_protocol": True,
                    "decode_speed_target": True,
                    "cb_system_prompt_working_set_pressure": True,
                    "source_vs_quant_first_divergence": True,
                    "mimo_media_wired": True,
                },
                "latest_decode_speed_evidence": {
                    "status": "pass",
                    "bundle_decode_tps": 42.0,
                    "greedy_decode_tps": 43.0,
                    "coherency_exact": True,
                    "generic_turboquant_kv_enabled": False,
                    "native_cache_type": "mixed_swa_kv",
                    "switchglu_fastpath_active": True,
                    "async_decode_wait_dominates": False,
                    "decode_bottleneck_classification": (
                        "speed_floor_cleared_with_fastpath"
                    ),
                },
                "blockers": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_vs_quant_path = root / CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT
    source_vs_quant_path.parent.mkdir(parents=True, exist_ok=True)
    source_vs_quant_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "remote_evidence_only": False,
                "source_model_path": "/Users/eric/models/MiMo-V2.5-source",
                "quant_model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
                "rows": [
                    {
                        "prompt_name": "short_exact_ack",
                        "classification": "source_and_quant_match",
                        "source_output": "ACK",
                        "quant_output": "ACK",
                        "first_divergence": None,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_truth_path = root / CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT
    metadata_truth_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_truth_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "runtime_modalities": ["text"],
                "preserved_modalities": ["vision", "audio"],
                "unwired_modalities": ["vision", "audio"],
                "multimodal_status": "weights_preserved_text_runtime",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    classifier_path = root / CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "classification": "exactness_release_cleared_by_stronger_evidence",
                "source_vs_quant_load_performed": False,
                "excluded_surfaces": {
                    "parser_argument_rewrite": True,
                    "prefix_paged_l2_or_kv_quant_primary_cause": True,
                    "hidden_stochastic_sampling_primary_cause": True,
                    "api_sampler_non_top1_selection": True,
                },
                "unresolved_surfaces": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    restart_l2_path = root / CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT
    restart_l2_path.parent.mkdir(parents=True, exist_ok=True)
    restart_l2_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "classification": {
                            "status": "pass",
                            "cache_status": "pass",
                            "output_status": "pass",
                            "checks": {
                                "restart_l2_restore_observed": True,
                                "second_cached_tokens_positive": True,
                                "block_disk_hit_positive": True,
                            },
                        }
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    responses_tools_path = root / CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT
    responses_tools_path.parent.mkdir(parents=True, exist_ok=True)
    responses_tools_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "proven": {
                    "responses_endpoint_used": True,
                    "responses_delta_streaming": True,
                    "responses_previous_response_id_tool_followup": True,
                    "cache": {
                        "cache_hit_tokens": 4552,
                        "l2_block_tokens_on_disk": 4960,
                    },
                },
                "red": {"long_tool_loop": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_open_mimo_v2_sink_ab_artifact(root: Path) -> None:
    path = root / CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "open",
                "normal_coherent": False,
                "sink_disabled_coherent": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _expected_passing_covered_live_smoke_summaries() -> dict[str, object]:
    return {
        "status": "pass",
        "non_mimo_status": "pass",
        "artifacts": {
            row_id: {
                "artifact": artifact,
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "failing_results": [],
            }
            for row_id, artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.items()
        },
        "missing": [],
        "not_pass": [],
        "non_mimo_missing": [],
        "non_mimo_not_pass": [],
    }


def _expected_passing_covered_live_tool_smoke_summaries() -> dict[str, object]:
    return {
        "status": "pass",
        "non_mimo_status": "pass",
        "artifacts": {
            row_id: {
                "artifact": artifact,
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "failing_results": [],
            }
            for row_id, artifact in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.items()
        },
        "missing": [],
        "not_pass": [],
        "non_mimo_missing": [],
        "non_mimo_not_pass": [],
    }


def _expected_passing_diagnostic_live_smoke_summaries() -> dict[str, object]:
    def expected_failures(expectation: dict[str, object]) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        for label, reason_or_reasons in expectation["expected_label_reasons"].items():
            reasons = (
                reason_or_reasons
                if isinstance(reason_or_reasons, list)
                else [reason_or_reasons]
            )
            failures.extend(
                {"label": str(label), "reason": str(reason)}
                for reason in reasons
            )
        return failures

    return {
        "status": "pass",
        "artifacts": {
            row_id: {
                "artifact": expectation["artifact"],
                "status": expectation["expected_status"],
                "completed": 1,
                "failed": 1,
                "matched_expected_failures": expected_failures(expectation),
                "unexpected_results": [],
            }
            for row_id, expectation in CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS.items()
        },
        "missing": [],
        "not_expected": [],
    }


def _passing_open_requirement_details() -> dict[str, object]:
    return {
        "Cross-family live multi-turn smoke matrix is release-cleared": {
            "details": {
                "missing_required_family_keys": [],
                "missing_evidence": [],
            }
        },
        "Real Electron UI cross-family live model matrix is release-cleared": {
            "details": {
                "release_boundary": "current_real_ui_matrix_is_authoritative_for_unblocked_non_mimo",
                "real_ui_live_model_matrix": {
                    "unblocked_non_mimo_status": "pass",
                    "unblocked_non_mimo_missing_families": [],
                    "unblocked_non_mimo_partial_families": [],
                    "covered_families": {"zaya_text": {"status": "pass"}},
                    "mixed_model_identity_families": [],
                },
                "required_real_ui_surfaces": [
                    "current Electron dev build",
                    "real loaded model per target family",
                    "chat settings persistence and non-sticky defaults",
                    "server cache setting controls",
                    "Responses and Chat Completions paths",
                    "long multi-turn tool calls",
                    "reasoning/tool parser display",
                    "prefix/cache hit telemetry",
                    "image/video follow-up where supported",
                    "no raw parser or reasoning tag leakage",
                ],
            }
        },
        "DSV4 long-output/code/file-generation quality is release-cleared": {
            "details": {
                "direct_off_exactness_boundary": {
                    "direct_off_failure_spans_chat_responses_and_completion": True,
                    "requested_thinking_success_is_diagnostic_only": True,
                    "requested_thinking_success_clears_direct_off": False,
                    "hidden_force_on_would_be_false_clearance": True,
                },
                "exact_code_root_boundary": {
                    "direct_off_route_wide_failure": True,
                    "requested_thinking_is_diagnostic_only": True,
                    "prefix_cache_not_sufficient_root_cause": True,
                    "forced_sampler_controls_not_sufficient_root_cause": True,
                    "bundle_defaults_do_not_clear_direct_off": True,
                    "direct_off_three_dot_identifier_boundary": True,
                    "current_primary_failure": "direct_off_exact_code_generation",
                },
                "current_source_full_output_preflight": {
                    "artifact": CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
                    "artifact_present": True,
                    "created_at": "2026-06-01T17:56:03-0700",
                    "status": "skipped",
                    "reason": "insufficient_free_memory",
                    "model": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520",
                    "commands": {
                        "memory": "vm_stat",
                        "memory_pressure": "memory_pressure",
                    },
                    "available_gb": 103.87,
                    "required_available_gb": 120.0,
                    "required_free_gb": 120.0,
                    "min_free_gb": 120.0,
                    "available_for_gate_gb": 103.87,
                    "memory_gap_gb": 16.13,
                    "strict_vm_stat_memory_gap_gb": 16.13,
                    "psutil_available_gap_gb": 16.13,
                    "memory_pressure_free_percent": 94,
                    "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                    "did_not_launch": True,
                    "launch_decision": "do_not_launch",
                    "launch_allowed": False,
                    "launch_blockers": ["insufficient_memory"],
                    "active_heavy_process_count": 0,
                    "active_heavy_processes": [],
                    "top_memory_processes": [
                        {
                            "pid": 1001,
                            "rss_gb": 10.0,
                            "command": "/Applications/vMLX.app/Contents/MacOS/vMLX",
                        }
                    ],
                    "case_count": 14,
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
                },
                "real_ui_dsv4_memory_preflight": {
                    "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
                    "status": "skipped",
                    "reason": "insufficient_memory",
                    "did_not_launch": True,
                    "launch_blockers": ["insufficient_memory"],
                },
            }
        },
    }


def _passing_current_suite_progress_checkpoints() -> dict[str, object]:
    steps = {
        "objective_digest": {"status": "pass"},
        "release_regression_manifest": {"status": "pass"},
    }
    return {
        "current_step": None,
        "completed_steps": list(steps),
        "steps": steps,
    }


def test_release_regression_manifest_covers_required_domains():
    manifest = build_manifest()
    domains = {row["domain"] for row in manifest["rows"]}

    assert REQUIRED_RELEASE_DOMAINS.issubset(domains)


def test_release_regression_manifest_has_stable_unique_ids():
    manifest = build_manifest()
    row_ids = [row["id"] for row in manifest["rows"]]

    assert len(row_ids) == len(set(row_ids))
    assert all(row_id == row_id.lower() for row_id in row_ids)
    assert all(" " not in row_id for row_id in row_ids)


def test_release_regression_manifest_artifacts_are_unique_per_row():
    manifest = build_manifest()

    for row in manifest["rows"]:
        artifacts = row["artifacts"]
        assert len(artifacts) == len(set(artifacts)), row["id"]


def test_release_regression_manifest_artifacts_are_concrete_files():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for artifact in row["artifacts"]:
            assert not artifact.endswith("/"), f"{row['id']} lists directory artifact {artifact}"


def test_release_regression_manifest_tracks_no_fake_sampler_policy():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    assert "generation-defaults-no-hidden-forcing" in rows
    assert rows["generation-defaults-no-hidden-forcing"]["domain"] == "generation_defaults"
    assert "generation_config.json" in " ".join(rows["generation-defaults-no-hidden-forcing"]["proves"])
    assert "jang_config.json" in " ".join(rows["generation-defaults-no-hidden-forcing"]["proves"])


def test_release_regression_manifest_tracks_new_chat_output_cap_inheritance_guard():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "new-chat" in joined
    assert "model-owned maxTokens" in joined
    assert "current-max-output-context-contract-20260522-chat-auto-server-default.json" in joined


def test_release_regression_manifest_tracks_server_chat_max_output_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "server startup maxTokens and chat maxTokens remain independent" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined
    assert "new chat output caps are not inherited or made sticky" in joined
    assert "below or above the server startup default" in joined
    assert "persisted chat maxTokens cannot relaunch server" in joined
    assert "current-max-output-context-contract-20260522-persisted-chat-output-cap.json" in joined
    assert "current-max-output-context-contract-20260522-new-chat-output-cap-nonsticky.json" in joined
    assert "current-max-output-context-contract-20260522-responses-output-boundary.json" in joined
    assert "current-max-output-context-contract-20260522-request-default-mutation.json" in joined
    assert "Streaming Chat/Responses output caps do not mutate the server startup default" in joined
    assert "current-max-output-context-contract-20260523-streaming-mutation.json" in joined
    assert "Responses maxTokens below or above the server startup default remain request scoped" in joined
    assert "Responses Auto does not synthesize max_output_tokens" in joined
    assert "DB-cleared NULL chat maxTokens stays Auto for Chat Completions and Responses" in joined
    assert "current-max-output-context-contract-20260523-null-chat-cap.json" in joined
    assert "do not mutate the server startup default" in joined
    assert "Raw API non-positive max_tokens/max_output_tokens are rejected" in joined
    assert "chat reset does not convert model max_new_tokens" in joined
    assert "string-shaped legacy session maxTokens values are cleared" in joined
    assert "current-max-output-context-contract-20260522-reset-policy-string-legacy.json" in joined
    assert "current-max-output-context-contract-20260522-api-validator-caps.json" in joined
    assert "API gateway output-budget and context-budget paths are source-hashed" in joined
    assert "DSV4 request-budget helper is source-hashed with the max-output boundary gate" in joined
    assert "Casual preset maxTokens is documented as an explicit server output cap" in joined
    assert "current-max-output-context-contract-20260522-casual-server-output-cap.json" in joined


def test_release_regression_manifest_tracks_multifamily_live_workflow_gate():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    for model_row in (
        "gemma4_crack",
        "minimax_m27_tq_k",
        "qwen36_moe_crack",
        "zaya_jangtq2",
        "zaya_vl_jangtq4",
        "ling_flash_tq",
        "nemotron_omni_tq2",
        "dsv4_jang_local",
    ):
        assert model_row in joined

    for contract in (
        "multi-turn",
        "tool",
        "reasoning parser",
        "prefix",
        "cache",
        "memory",
        "wrong-language",
        "generation_config.json",
        "jang_config.json",
        "VL",
        "video",
        "Chat Completions",
        "Responses",
        "Anthropic",
        "Ollama",
        "stop/disconnect",
        "full visible output",
        "full reasoning output",
        "tail review",
    ):
        assert contract in joined

    assert "revalidates visible text for unexpected CJK" in joined
    assert "--include-tools" in joined
    assert "tool_choice=required" in joined
    assert "current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json" in joined
    assert "current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json" in joined
    assert (
        "current-all-local-model-smoke-zaya-vl-jangtq4-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-mxfp4-bundled-after-source-tool-pass-20260607/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-mxfp4-bundled-after-source-tool-pass-20260607/summary.json"
        in joined
    )
    assert "ZAYA text MXFP4 tool probe remains diagnostic" in joined
    assert "ZAYA-VL JANGTQ4 true-bundled tool probe passes tool_required" in joined
    assert (
        "ZAYA-VL JANGTQ4 true-bundled media-sentinel smoke passes ACK/cache, tool calls, and VL colors"
        in joined
    )
    assert (
        "2026-05-26 reasoning-enabled rerun is diagnostic-fail because reasoning_on produced reasoning-only empty visible output"
        in joined
    )
    assert (
        "prompt-sensitive no-media control returned image"
        in joined
    )
    assert "semantic text-after-image isolation" not in joined
    assert "ZAYA-VL JANGTQ4 bundled smoke reproduces the same exact ACK/template failure" not in joined
    assert "Nemotron Omni Nano failed prompt variants remain diagnostic" in joined
    assert "so nemotron remains attempted-but-not-covered" not in joined
    assert (
        "current-all-local-model-smoke-nemotron-omni-jangtq-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-nemotron-omni-jangtq-explicit-nomedia-bundled-20260524/summary.json"
        in joined
    )
    assert "current-filtered-live-smoke-nemotron-omni-jangtq-20260607/summary.json" in joined
    assert "Nemotron Omni Nano bundled tool probe passes tool_required" in joined
    assert "current-nemotron-omni-no-media-prompt-variants-20260524/result.json" in joined
    assert "current-nemotron-omni-no-media-system-prompt-diagnostic-20260524/result.json" in joined
    assert (
        "current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-filtered-live-smoke-ling-flash-jangtq-20260607/summary.json"
        in joined
    )
    assert "Ling/Bailing bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-gemma26-jang4m-bundled-tools-media-20260607/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-toolprobe-20260525/summary.json"
        not in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.values()
    )
    assert "Gemma4 bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/summary.json"
        in joined
    )
    assert "Qwen3.6 MXFP4 bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-hy3-jangtq2-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-ling-hy3-nemotron-tools-media-20260606/summary.json"
        in joined
    )
    assert "Hy3 bundled tool probe passes tool_required" in joined
    assert (
        "current-all-local-model-smoke-minimax-small-jangtq-bundled-20260524/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-minimaxk-bundled-after-required-tool-256-20260607/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-minimaxk-bundled-after-required-tool-256-20260607/summary.json"
        in joined
    )
    assert "MiniMax bundled tool probe passes required record_fact" in joined
    assert (
        "current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json"
        in joined
    )
    assert (
        "current-all-local-model-smoke-zaya-vl-mxfp4-bundled-after-source-tool-pass-20260607/summary.json"
        in joined
    )
    assert "ZAYA text MXFP4 bundled tool probe remains a diagnostic failure" in joined
    assert "--only DeepSeek-V4-Flash-JANGTQ-K" in joined
    assert (
        "current-filtered-live-smoke-dsv4-jangtq-k-20260607/summary.json"
        in joined
    )
    assert "--only ZAYA1-VL-8B-MXFP4" in joined
    assert "--only ZAYA1-VL-8B-JANGTQ4" in joined
    assert "--only Nemotron-Omni-Nano-JANGTQ-CRACK" in joined
    assert "--only Ling-2.6-flash-JANGTQ" in joined
    assert "--only Gemma-4-26B-A4B-it-JANG_4M-CRACK" in joined
    assert "--only Qwen3.6-27B-MXFP4-CRACK" in joined
    assert "--only Hy3-preview-JANGTQ2" in joined
    assert "--only MiniMax-M2.7-Small-JANGTQ" in joined


def test_release_regression_manifest_tracks_covered_live_smoke_artifacts():
    artifacts = set(CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.values())

    assert (
        "build/current-filtered-live-smoke-ling-flash-jangtq-20260607/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-gemma26-jang4m-bundled-tools-media-20260607/summary.json"
        in artifacts
    )
    assert (
        "build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/summary.json"
        in artifacts
    )
    assert (
        "build/current-filtered-live-smoke-dsv4-jangtq-k-20260607/summary.json"
        in artifacts
    )
    assert (
        "build/current-filtered-live-smoke-nemotron-omni-jangtq-20260607/summary.json"
        in artifacts
    )


def test_release_regression_manifest_tracks_current_post_budget_edge_proof_sweep():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    for row_id, artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.items():
        assert artifact in rows[row_id]["artifacts"], row_id


def test_release_regression_manifest_current_sweep_uses_latest_live_smoke_artifacts():
    current_artifacts = [
        CURRENT_REGRESSION_SUITE_ARTIFACT,
        CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
        *CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values(),
    ]
    manifest = build_manifest()
    row_text = json.dumps(manifest["rows"])
    joined = "\n".join(current_artifacts)

    assert "current-regression-suite-20260528-installed-aggregate-stale.json" not in joined
    assert "current-regression-suite-20260528-epipe-aggregate-guard.json" not in joined
    assert "current-regression-suite-20260528-dsv4-continue-refresh.json" not in joined
    assert "current-regression-suite-after-pr-intake-matrix-refresh-20260609.json" in joined
    assert "current-regression-suite-after-structured-schema-decode-20260609.json" not in joined
    assert "current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json" in joined
    assert "current-noheavy-api-cache-contract-after-structured-schema-decode-20260609.json" not in joined
    assert "current-regression-suite-gemma4-release-boundary-after-ui-e2e-fixes-dmg-build-20260604.json" not in joined
    assert "current-regression-suite-20260602-v1553-installed-tahoe-refresh.json" not in joined
    assert "current-regression-suite-20260602-vm-stat-gate-validation.json" not in joined
    assert "current-regression-suite-20260601-pipe-safe-runner.json" not in joined
    assert "current-regression-suite-20260601-developer-id-dmg-assertions.json" not in joined
    assert "current-regression-suite-20260601-cache-ipc-installed-refresh.json" not in joined
    assert "current-regression-suite-20260601-after-adhoc-reseal.json" not in joined
    assert "current-regression-suite-20260601-qwen3vl-minicpm-mpp-final.json" not in joined
    assert "current-regression-suite-20260531-live-epipe-signing-dsv4-refresh.json" not in joined
    assert "current-regression-suite-20260531-step37-ui-pagedlock.json" not in joined
    assert "current-regression-suite-20260531-step37-mixed-swa-runtime.json" not in joined
    assert "current-regression-suite-20260531-packaged-pointer-bundle-refresh.json" not in joined
    assert "current-regression-suite-20260531-step37-integrated-tool-l2-proof.json" not in joined
    assert "current-regression-suite-20260531-two-turn-responses-delta-gate.json" not in joined
    assert "current-regression-suite-20260531-live-chat-tools-proof-refresh.json" not in joined
    assert "current-regression-suite-20260531-childstream-epipe-guard.json" not in joined
    assert "current-regression-suite-20260530-bundled-sync-step37-projector-rerun.json" not in joined
    assert "current-regression-suite-20260530-bundled-sync-step37-projector.json" not in joined
    assert "current-regression-suite-20260529-step37-text-bridge.json" not in joined
    assert "current-regression-suite-20260528-prepackage-gate.json" not in joined
    assert "current-regression-suite-20260528-release-ready-dmg-gate.json" not in joined
    assert "current-regression-suite-20260528-release-gate-wired.json" not in joined
    assert "current-regression-suite-20260528-release-ready-top-level.json" not in joined
    assert "current-regression-suite-20260528-dsv4-memory-refresh.json" not in joined
    assert "current-regression-suite-20260528-signing-detail-ledger.json" not in joined
    assert "current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json" in joined
    assert "current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json" in row_text
    assert "run_installed_app_runtime_parity_audit.py --app panel/release/sequoia-app/mac-arm64/vMLX.app --out build/current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json" in row_text
    assert "current-installed-app-runtime-parity-audit-20260602-developer-id-installed-signing.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260602-developer-id-installed-signing.json" not in row_text
    assert "current-installed-app-runtime-parity-audit-20260602-performance-health-epipe.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260602-performance-health-epipe.json" not in row_text
    assert "current-installed-app-runtime-parity-audit-20260601-epipe-renderer-installed.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260601-epipe-renderer-installed.json" not in row_text
    assert "current-installed-app-runtime-parity-audit-20260531-live-epipe-refresh.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260531-live-epipe-refresh.json" not in row_text
    assert "current-installed-app-runtime-parity-audit-20260531-childstream-epipe-installed-sync.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260531-childstream-epipe-installed-sync.json" not in row_text
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-aggregate-guard.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-aggregate-guard.json" not in row_text
    assert "current-installed-app-runtime-parity-audit-tahoe-checkpoint-dmg-20260609.json" in joined
    assert "current-staged-app-runtime-parity-audit-20260602-performance-health-epipe.json" not in joined
    assert "current-staged-app-runtime-parity-audit-20260601-cache-ipc-epipe-staged.json" not in joined
    assert "current-staged-app-runtime-parity-audit-20260601-wrapper-epipe-package-refresh.json" not in joined
    assert "current-staged-app-runtime-parity-audit-20260531-step37-mixed-swa-runtime.json" not in joined
    assert "current-staged-app-runtime-parity-audit-20260528-staged-runtime-recheck.json" not in joined
    assert "current-staged-app-runtime-parity-audit-20260528-installed-aggregate-stale.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-postrefresh.json" not in joined
    assert "current-installed-app-runtime-parity-audit-20260528-epipe-diagnostic-scan.json" not in joined
    assert "current-regression-suite-20260528-epipe-diagnostic-scan.json" not in joined
    assert "current-regression-suite-20260528-named-family-cache-matrix.json" not in joined
    assert "current-regression-suite-20260528-dsv4-blocker-evidence.json" not in joined
    assert "current-regression-suite-20260528-ollama-embedding-timeout.json" not in joined
    assert "current-regression-suite-20260528-local-release-evidence.json" not in joined
    assert "current-regression-suite-20260528-release-clearance-open.json" not in joined
    assert "current-regression-suite-20260528-api-ollama-embeddings-single-model.json" not in joined
    assert "current-regression-suite-20260528-dsv4-legacy-clearance-cleanup.json" not in joined
    assert "current-regression-suite-20260528-signing-root-cause.json" not in joined
    assert "current-regression-suite-20260528-current-cohesive-audit.json" not in joined
    assert "current-regression-suite-20260528-nonmimo-zaya-direct.json" not in joined
    assert "current-regression-suite-20260528-ollama-chat-epipe-dsv4-preflight.json" not in joined
    assert "current-regression-suite-20260528-ollama-chat-epipe-guard.json" not in joined
    assert "current-regression-suite-20260528-dsv4-0625-preflight.json" not in joined
    assert "current-regression-suite-20260528-current-live-smoke-pointers.json" not in joined
    assert "current-regression-suite-20260528-dsv4-0600-preflight.json" not in joined
    assert "current-regression-suite-20260528-userdata-epipe-scan.json" not in joined
    assert "current-regression-suite-20260528-dsv4-preflight-postinstall.json" not in joined
    assert "current-regression-suite-20260528-installed-ipc-epipe-guard.json" not in joined
    assert "current-regression-suite-20260528-ipc-epipe-request-guard.json" not in joined
    assert "current-regression-suite-20260528-dsv4-identifier-candidates.json" not in joined
    assert "current-regression-suite-20260528-issue175-179-boundary-pass.json" not in joined
    assert "current-regression-suite-20260528-epipe-request-end-guard.json" not in joined
    assert "current-regression-suite-20260528-issue179-local-repro-clean.json" not in joined
    assert "current-regression-suite-20260528-issue179-reporter-shape-guard.json" not in joined
    assert "current-regression-suite-20260528-epipe-socket-guard.json" not in joined
    assert "current-regression-suite-20260528-generation-family-matrix.json" not in joined
    assert "current-regression-suite-20260528-cache-swa-hca-csa-row.json" not in joined
    assert "current-regression-suite-20260528-dsv4-memory-gap-preflight.json" not in joined
    assert "current-regression-suite-20260528-ollama-proxy-error-guard.json" not in joined
    assert "current-regression-suite-20260528-reporter-parity-collector.json" not in joined
    assert "current-regression-suite-20260528-reporter-parity-dsv4-preflight-refresh.json" not in joined
    assert "current-regression-suite-20260528-ollama-epipe-multiline-installed.json" not in joined
    assert "current-regression-suite-20260528-issue179-installed-session-proof.json" not in joined
    assert "current-regression-suite-20260528-issue179-launch-config-proof.json" not in joined
    assert "current-regression-suite-20260528-issue179-request-shape-proof.json" not in joined
    assert "current-regression-suite-20260528-issue179-installed-launcher-proof.json" not in joined
    assert "current-regression-suite-20260528-dsv4-preflight-refresh.json" not in joined
    assert "current-regression-suite-20260528-admin-sleep-sourcehash.json" not in joined
    assert "current-regression-suite-20260528-issue179-econnreset-boundary.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-after-lfm-step-manifest-fix-20260604.json" in joined
    assert "current-real-ui-dsv4-memory-preflight-20260603-second-local-check.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260602-developer-id-local-recheck.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260601-local-recheck.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260601-post-wrapped-epipe.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260601-local-refresh.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260531-live-refresh.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260530-local-refresh.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-continue-refresh.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-epipe-install.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-0625-recheck.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-0600-recheck.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528-postinstall.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260528.json" not in joined
    assert "current-real-ui-dsv4-memory-preflight-20260527.json" not in joined
    assert "current-regression-suite-20260527-local-only-mimo-boundary.json" not in joined
    assert "current-regression-suite-20260527-after-proof-source-hash-coverage.json" not in joined
    assert "current-regression-suite-20260527-after-think-xml-registry-fix.json" not in joined
    assert "current-regression-suite-20260527-issue179-real-ui-diagnostic-manifest.json" not in joined
    assert "current-regression-suite-20260527-live-matrix-audit-final.json" not in joined
    assert "current-regression-suite-20260527-mimo-v2-jang2l-noheavy-proof-rerun.json" not in joined
    assert "current-regression-suite-20260527-issues-175-178-bundled-sync.json" not in joined
    assert "current-regression-suite-20260527-zaya-vl-point-leak-proof-tightening.json" not in joined
    assert "current-regression-suite-20260527-zaya-vl-cachecontrols.json" not in joined
    assert "current-regression-suite-20260527-gemma4-speed-overclaim-fixed.json" not in joined
    assert "current-regression-suite-20260527-real-ui-family-expansion.json" not in joined
    assert "current-regression-suite-20260527-real-ui-zaya-vl-image.json" not in joined
    assert "current-regression-suite-20260526-real-ui-live-model-slice.json" not in joined
    assert "current-regression-suite-20260526-dev-ui-wiring.json" not in joined
    assert "current-regression-suite-20260526-settings-audit.json" not in joined
    assert "current-regression-suite-20260525-smoke-reasoning-loop-guard.json" not in joined
    assert "current-regression-suite-20260525-gemma-short-nocache-boundary.json" not in joined
    assert "current-regression-suite-20260525-gemma-internal-speed-gate.json" not in joined
    assert "current-regression-suite-20260525-dsv4-copy-block-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-prompt-guard-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-nocache-ab-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-default-cache-raw-boundary.json" not in joined
    assert "current-regression-suite-20260525-dsv4-bundle-defaults-cjk-wide.json" not in joined
    assert "current-regression-suite-20260525-dsv4-dryrun-policy-wired.json" not in joined
    assert "current-regression-suite-20260525-cjk-smoke-guard.json" not in joined
    assert "current-regression-suite-20260525-gemma-installed-speed-boundary.json" not in joined
    assert "current-regression-suite-20260524-openai-single-model-streaming-audit.json" not in joined
    assert "current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json" in joined
    assert "current-api-surface-contract-20260602-cache-detail-zero-cached.json" not in joined
    assert "current-api-surface-contract-20260531-nested-epipe-childstream-refresh.json" not in joined
    assert "current-api-surface-contract-20260529-single-model-transition-lock.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-embedding-timeout.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-embeddings-single-model.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-nested-disconnect.json" not in joined
    assert "current-api-surface-contract-20260528-cache-endpoints-autoswitch-expanded.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-client-close-epipe-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ipc-epipe-request-guard.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-request-end-guard.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-socket-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-proxy-error-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-epipe-multiline-guard.json" not in joined
    assert "current-api-surface-contract-20260527-cache-endpoint-autoswitch-proof.json" not in joined
    assert "current-api-surface-contract-20260526-single-model-auto-switch-review.json" not in joined
    assert "current-api-surface-contract-20260525-single-model-responses-deltas.json" not in joined
    assert "current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json" in joined
    assert "current-packaged-integrity-contract-20260601-qwen-fix-resigned-staged-app.json" not in joined
    assert "current-packaged-integrity-contract-20260601-developer-id-dmg-assertions.json" not in joined
    assert "current-packaged-integrity-contract-20260601-cache-ipc-epipe-package-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260531-after-lfm2-staged-sync.json" not in joined
    assert "current-packaged-integrity-contract-20260531-step37-mixed-swa-runtime.json" not in joined
    assert "current-packaged-integrity-contract-20260531-after-adhoc-reseal.json" not in joined
    assert "current-packaged-integrity-contract-20260531-local-release-decision-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260531-live-signing-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260531-gemma4-l2-rotating-kv-fix.json" not in joined
    assert "current-packaged-integrity-contract-20260531-childstream-epipe-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260530-bundled-sync-after-step37-projector.json" not in joined
    assert "current-packaged-integrity-contract-20260529-step37-text-bridge.json" not in joined
    assert "current-packaged-integrity-contract-20260528-installed-aggregate-stale.json" not in joined
    assert "current-packaged-integrity-contract-20260528-prepackage-gate.json" not in joined
    assert "current-packaged-integrity-contract-20260528-release-ready-dmg-gate.json" not in joined
    assert "current-packaged-integrity-contract-20260528-release-ready-gate.json" not in joined
    assert "current-packaged-integrity-contract-20260528-signing-keychain-root-cause.json" not in joined
    assert "current-packaged-integrity-contract-20260528-signing-root-cause.json" not in joined
    assert "current-packaged-integrity-contract-20260528-ollama-chat-epipe-guard.json" not in joined
    assert "current-packaged-integrity-contract-20260528-ipc-epipe-request-guard.json" not in joined
    assert "current-packaged-integrity-contract-20260528-epipe-socket-guard-installed-sync.json" not in joined
    assert "current-packaged-integrity-contract-20260527-after-think-xml-registry-fix-rerun.json" not in joined
    assert "current-packaged-integrity-contract-20260527-issues-175-178-bundled-sync-rerun.json" not in joined
    assert "current-packaged-integrity-contract-20260527-mimo-v2-jang2l-bundled-import-rerun.json" not in joined
    assert "current-packaged-integrity-contract-20260526-bundled-release-proof.json" not in joined
    assert "current-packaged-integrity-contract-20260525-additional-args-guard.json" not in joined
    assert "current-regression-suite-20260524-crossfamily-cleared-dsv4-open.json" not in joined
    assert "current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json" in joined
    assert "current-generation-defaults-contract-20260602-step-greedy-display.json" not in joined
    assert "current-generation-defaults-contract-20260531-post-step-lfm-refresh.json" not in joined
    assert "current-generation-defaults-contract-20260526-settings-audit.json" not in joined
    assert "current-max-output-context-contract-after-jangtq2-objective-refresh-20260607.json" in joined
    assert "current-max-output-context-contract-20260526-settings-audit.json" not in joined
    assert "current-reasoning-template-contract-20260526-settings-audit.json" in joined
    assert "current-generation-defaults-contract-20260525-additional-args-guard.json" not in joined
    assert "current-max-output-context-contract-20260524-after-gemma4-telemetry-final.json" not in joined
    assert "current-reasoning-template-contract-20260523-post-budget-edge.json" not in joined
    assert "current-regression-suite-20260524-gemma4-visible-crossfamily-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-live-smoke-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-gemma4-smoke-dsv4-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-after-gemma4-mixed-swa-telemetry.json" not in joined
    assert "current-regression-suite-20260524-gemma4-visible-open.json" not in joined
    assert "current-regression-suite-20260524-after-dsv4-neutral-reppen.json" not in joined
    assert "current-generation-defaults-contract-20260524-after-dsv4-neutral-reppen.json" not in joined
    assert "current-max-output-context-contract-20260524-after-dsv4-neutral-reppen.json" not in joined
    assert "current-regression-suite-20260524-default-cache-live-review.json" not in joined
    assert "current-regression-suite-20260524-zaya-vl-smoke-open.json" not in joined
    assert "current-regression-suite-20260524-nemotron-smoke-open.json" not in joined
    assert "current-packaged-integrity-contract-20260524-default-cache-live-review.json" not in joined
    assert "current-generation-defaults-contract-20260524-after-ling-topk-policy.json" not in joined
    assert "current-max-output-context-contract-20260524-after-chatmax-dsv4-server-edits.json" not in joined


def test_release_regression_manifest_tracks_electron_dev_ui_wiring_proof():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["electron-dev-ui-tools-reasoning-wiring"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["mode"] == "dev-ui"
    assert "VMLX_LIVE_PROOF_BASENAME=2026-05-31-live-chat-tools-reasoning" in joined
    assert "panel/scripts/live-chat-tools-reasoning-proof.mjs" in joined
    assert "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-proof.json" in joined
    assert "mock-server UI wiring proof only" in joined
    assert "does not clear live-model language quality" in joined
    assert "reasoning segments" in joined
    assert "read_image/read_video" in joined
    assert "Block Disk Cache L2" in joined


def test_release_regression_manifest_tracks_real_electron_ui_live_model_slice():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["electron-dev-ui-real-model-live-slice"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["mode"] == "dev-ui-live-model"
    assert "panel/scripts/live-real-ui-model-proof.mjs" in joined
    assert "VMLINUX_REAL_UI_MODEL_PATH" in joined
    assert "ZAYA1-8B-MXFP4" in joined
    assert "real vmlx_engine server" in joined
    assert "window.api.chat.sendMessage" in joined
    assert "not enough to clear the full cross-family real-UI blocker" in joined
    assert "current-real-ui-live-model-zaya-text-20260526-proof.json" in joined
    assert "Ling-2.6-flash-JANGTQ" in joined
    assert "current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json" in joined
    assert "Gemma-4-26B-A4B-it-JANG_4M-CRACK" in joined
    assert "current-real-ui-live-model-gemma4-26b-jang4m-20260526-proof.json" in joined
    assert (
        "diagnostic-real-ui-live-model-gemma4-responses-tools-reasoning-20260527-proof.json"
        in joined
    )
    assert "Responses/tools/reasoning" in joined
    assert "mixed-SWA paged cache" in joined
    assert "affine MLX matmul" in joined
    assert "80 tok/s" in joined
    assert "Hy3-preview-JANGTQ2" in joined
    assert "current-real-ui-live-model-hy3-jangtq2-20260526-proof.json" in joined
    assert "TurboQuantKVCache/paged prefix cache" in joined
    assert "current-lfm25-direct-responses-runcommand-cache-boundary-20260530.json" in joined
    assert "server-clean while the real UI stream path remains malformed" in joined


def test_release_regression_manifest_real_ui_live_model_script_exists_and_uses_real_session_path():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "VMLINUX_REAL_UI_MODEL_PATH" in source
    assert "VMLINUX_REAL_UI_WIRE_API" in source
    assert "VMLINUX_REAL_UI_BUILTIN_TOOLS" in source
    assert "real_ui_tool_probe_1.txt" in source
    assert "real_ui_tool_probe_2.txt" in source
    assert "REAL_UI_LIVE_TOOL_ONE" in source
    assert "REAL_UI_LIVE_TOOL_TWO" in source
    assert "VMLINUX_REAL_UI_WORKING_DIRECTORY" in source
    assert "VMLINUX_REAL_UI_ENABLE_THINKING" in source
    assert "VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS" in source
    assert "VMLINUX_REAL_UI_CHECK_MEDIA" in source
    assert "VMLINUX_REAL_UI_IMAGE_DATA_URL" in source
    assert "VMLINUX_REAL_UI_IMAGE_EXPECT_REGEX" in source
    assert "VMLINUX_REAL_UI_CHECK_VIDEO" in source
    assert "VMLINUX_REAL_UI_VIDEO_DATA_URL" in source
    assert "VMLINUX_REAL_UI_VIDEO_EXPECT_REGEX" in source
    assert "provenSurfaces" in source
    assert "serverCacheControls" in source
    assert "sectionButtons = [...document.querySelectorAll('button')]" in source
    assert "const normalized = text.replace(/\\\\s+/g, ' ').trim()" in source
    assert "titleWithoutDisclosure === title" in source
    assert "new MouseEvent('click'" in source
    assert "sectionClickResults" in source
    assert "media: mediaEvidence" in source
    assert "imageVerified" in source
    assert "videoVerified" in source
    assert "imageSemanticVerified" in source
    assert "videoSemanticVerified" in source
    assert "extractLiveSpeedSamples" in source
    assert "liveSpeedSamples" in source
    assert "live_speed_floor" in source
    assert "sendErrors" in source
    assert "rendererFailureStage" in source
    assert "status: rendererResult.rendererFailureStage ? 'fail' : 'pass'" in source
    assert "status: rendererResult.rendererFailureStage ? 'fail' : undefined" not in source
    assert "failureStage: rendererResult.rendererFailureStage || undefined" in source
    assert "persistedToolsByMessage" in source
    assert "persistedReasoningByMessage" in source
    assert "persistedReasoningText" in source
    assert "reasoningRawParserTagLeak" in source
    assert "reasoningCjkLeakCount" in source
    assert "reasoningKoreanLeakCount" in source
    assert "reasoningNumericRunCount" in source
    assert "numeric/list-like garbage leaked into reasoning segments" in source
    assert "--enable-auto-tool-choice" in source
    assert "--tool-call-parser" in source
    assert "window.api.sessions.create" in source
    assert "window.api.sessions.start" in source
    assert "window.api.chat.sendMessage" in source
    assert "window.api.performance.health" in source
    assert "window.api.cache.stats" in source
    assert "native_cache_status" in source
    assert "cache_endpoint_stats" in source
    assert "health.native_cache" in source
    assert "block_disk_cache" in source
    assert "cache_totals" in source
    assert "VMLINUX_REAL_UI_APP_PATH" in source
    assert "installedAppPath" in source
    assert "uiLaunchMode" in source
    assert "Contents', 'MacOS', 'vMLX'" in source
    assert "installed_app_ui" in source
    assert "terminateProcess" in source
    assert "childProcessTree" in source
    assert "terminateProcessTree" in source
    assert "process.kill(pid, signal)" in source
    assert "removeTemporaryTree(userDataDir)" in source
    assert "removeTemporaryTree(runDir)" in source
    assert "maxRetries" in source
    assert "ENOTEMPTY" in source
    assert "rawParserTagLeak" in source
    assert "<\\\\|point_start\\\\|>" in source
    assert "<\\\\|point_end\\\\|>" in source
    assert "cjkLeakCount" in source
    assert "clearTimeout(timer)" in source
    assert "current-real-ui-live-model-zaya-text-20260526" in source
    assert "server process exited before health" in source
    assert "server.proc.exitCode" in source
    assert "server.logs.slice(-80)" in source


def test_release_regression_manifest_real_ui_live_model_cdp_write_guards_epipe():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "function isSocketDisconnectError" in source
    assert "write EPIPE" in source
    assert "ERR_STREAM_DESTROYED" in source
    assert "writeClientFrame(payload)" in source
    assert "if (this.socket.destroyed || this.closed)" in source
    assert "this.rejectPending(error)" in source


def test_release_regression_manifest_live_chat_tools_proof_guards_epipe_writes():
    script = Path("panel/scripts/live-chat-tools-reasoning-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "function isSocketDisconnectError" in source
    assert "write EPIPE" in source
    assert "ERR_STREAM_DESTROYED" in source
    assert "function safeHttpWrite" in source
    assert "safeHttpWrite(res, 'data: [DONE]\\n\\n')" in source
    assert "writeClientFrame(payload)" in source
    assert "if (this.socket.destroyed || this.closed)" in source
    assert "this.rejectPending(error)" in source


def test_release_regression_manifest_real_ui_cache_checker_is_model_specific():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "VMLINUX_REAL_UI_CACHE_EXPECT_REGEX" in source
    assert "VMLINUX_REAL_UI_EXPECT_PAGED_CACHE_LOCKED" in source
    assert "cacheExpectRegex" in source
    assert "expectPagedCacheLocked" in source
    assert "new RegExp(cacheExpectRegex, 'i').test(bodyText)" in source

    verified_block = source.split("const verified = ", 1)[1].split("return {", 1)[0]
    assert "ZAYA typed CCA cache requires paged cache" not in verified_block
    assert "This architecture requires native/paged cache" not in verified_block


def test_release_regression_manifest_uses_installed_app_nemotron_cachecontrols_proof():
    row = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS[
        "nemotron_omni_nano_responses_tools_reasoning_cachecontrols"
    ]

    expected = (
        "current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601"
    )
    assert expected in row["proof"]
    assert expected in row["chat_screenshot"]


def test_release_regression_manifest_uses_installed_app_lfm25_responses_delta_cache_proof():
    row = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS["lfm25_moe_a1b_responses_delta"]

    expected = (
        "current-real-ui-live-model-lfm25-mxfp4-responses-tools-"
        "cachecontrols-20260607"
    )
    assert expected in row["proof"]
    assert expected in row["chat_screenshot"]


def test_release_regression_manifest_real_ui_script_preserves_tool_probe_file_evidence():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "const toolProbeFiles = {}" in source
    assert "function cacheReconstructionClean(result)" in source
    assert "worker-side paged cache reconstruction failed" in source
    result_block = source.split("const result = {", 2)[2].split("result.provenSurfaces", 1)[0]
    assert result_block.index("...rendererResult") < result_block.index("toolProbeFiles")
    semantics_block = source.split("function namedToolProbeSemanticsOk(result)", 1)[1].split(
        "function countMatches",
        1,
    )[0]
    assert "real_ui_tool_probe_2.txt" in semantics_block
    assert "REAL_UI_LIVE_TOOL_TWO" in semantics_block
    assert "&& fileSemanticsOk" in semantics_block


def test_release_regression_manifest_real_ui_script_records_stream_text_trace():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "streamTraceByMessage" in source
    assert "lastFullContent" in source
    assert "lastReasoningContent" in source
    assert "streamTrace: rendererResult.streamTraceByMessage" in source
    assert "requested Responses API mode but proof did not record responses_delta_streaming surface" in source


def test_release_regression_manifest_real_ui_script_requires_responses_cache_detail_when_requested():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "function responsesCacheDetailUsageSeen(result)" in source
    assert "responses_cache_detail_usage" in source
    assert (
        "requested Responses API cache controls but proof did not record responses_cache_detail_usage surface"
        in source
    )


def test_release_regression_manifest_real_ui_script_requires_generation_defaults_proof():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "generationDefaultsAppliedSeen(result)" in source
    assert "generation_defaults_applied" in source
    assert (
        "live proof did not record model-owned generation defaults / request max_tokens resolution"
        in source
    )


def test_release_regression_manifest_real_ui_script_requires_language_leak_surface():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "language_leak_check" in source
    assert "live proof did not record clean visible/reasoning language leak check" in source


def test_release_regression_manifest_real_ui_script_requires_clean_cache_hit_surface():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "cache_hit_telemetry" in source
    assert "cacheReconstructionClean(result)" in source
    assert "live proof did not record clean cache-hit telemetry" in source


def test_release_regression_manifest_chat_ipc_stream_metrics_include_cache_reuse_detail():
    source = Path("panel/src/main/ipc/chat.ts").read_text(encoding="utf-8")

    stream_sends = source.split('win.webContents.send("chat:stream",')[1:]
    assert stream_sends, "chat IPC stream emissions not found"
    for send_block in stream_sends:
        metrics_block = send_block.split("metrics: {", 1)[1].split("}", 1)[0]
        assert "cachedTokens" in metrics_block
        assert "cacheDetail" in metrics_block

    complete_block = source.split('win.webContents.send("chat:complete",', 1)[
        1
    ].split("metrics: {", 1)[1].split("}", 1)[0]
    assert "cachedTokens" in complete_block
    assert "cacheDetail" in complete_block
    assert "respUsage.input_tokens_details.cache_detail" in source


def test_release_regression_manifest_chat_ipc_rolls_back_failed_media_turns():
    source = Path("panel/src/main/ipc/chat.ts").read_text(encoding="utf-8")

    assert "rolled_back_failed_media_user_message" in source
    assert "rolled_back_empty_warning_media_user_message" in source
    assert "VLM image prefill guard" in source
    assert "db.deleteMessage(userMessage.id)" in source
    assert "hasMediaAttachments && !hadVisibleActivity && !wasAborted" in source
    assert "mediaWarningWithoutVisibleActivity" in source


def test_release_regression_manifest_real_ui_script_records_request_contract():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    result_block = source.split("const result = {", 2)[2].split(
        "result.visibleAssistantTurnsComplete",
        1,
    )[0]
    assert "requestContract" in result_block
    assert "promptOne" in result_block
    assert "promptTwo" in result_block
    assert "requestMaxTokens" in result_block
    assert "requestMaxPromptTokens" in result_block
    assert "maxToolIterations" in result_block
    assert "toolResultMaxChars" in result_block
    assert "imageExpectRegex" in result_block
    assert "videoExpectRegex" in result_block
    assert "cacheExpectRegex" in result_block


def test_release_regression_manifest_real_ui_script_records_request_contract_on_failure():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    failure_block = source.split("const result = {", 2)[1].split(
        "result.visibleAssistantTurnsComplete",
        1,
    )[0]
    assert "failureStage: 'renderer_real_ui_chat'" in failure_block
    assert "requestContract" in failure_block
    assert "promptOne" in failure_block
    assert "promptTwo" in failure_block
    assert "requestMaxTokens" in failure_block
    assert "requestMaxPromptTokens" in failure_block
    assert "maxToolIterations" in failure_block
    assert "toolResultMaxChars" in failure_block
    assert "wireApi" in failure_block
    assert "builtinToolsEnabled" in failure_block
    assert "enableThinking: enableThinkingOverride ?? null" in failure_block
    assert "checkServerCacheControls" in failure_block
    assert "checkMedia" in failure_block
    assert "checkVideo" in failure_block
    assert "expectPagedCacheLocked" in failure_block
    assert "imageExpectRegex" in failure_block
    assert "videoExpectRegex" in failure_block
    assert "cacheExpectRegex" in failure_block


def test_release_regression_manifest_real_ui_script_waits_for_async_l2_cache_stats():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "function l2DiskStorageSeen(cache)" in source
    assert "waitForCacheEndpointStorage" in source
    assert "expected cache endpoint L2 disk storage telemetry" in source
    renderer_block = source.split("const cacheAfter = await window.api.cache.stats", 1)[1].split(
        "const messages = await window.api.chat.getMessages",
        1,
    )[0]
    assert "const waitForCacheEndpointStorage = async (initial, sessionId)" in source
    assert "window.api.cache.stats(endpoint, sessionId)" in source
    assert "await waitForCacheEndpointStorage(cacheAfter, remote.session.id)" in renderer_block


def test_release_regression_manifest_real_ui_default_image_fixture_has_valid_png_crc():
    import base64
    import binascii
    import struct

    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")
    match = re.search(r"\|\| 'data:image/png;base64,([^']+)'", source)
    assert match, "default real UI image data URL not found"

    png = base64.b64decode(match.group(1), validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    offset = 8
    seen_iend = False
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        chunk_type = png[offset + 4 : offset + 8]
        chunk_data = png[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(
            ">I",
            png[offset + 8 + length : offset + 12 + length],
        )[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        assert actual_crc == expected_crc, f"invalid PNG CRC for {chunk_type!r}"
        offset += 12 + length
        if chunk_type == b"IEND":
            seen_iend = True
            break

    assert seen_iend
    assert offset == len(png)


def test_release_regression_manifest_real_ui_script_serializes_unset_thinking_contract():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    result_block = source.split("const result = {", 2)[2].split(
        "result.visibleAssistantTurnsComplete",
        1,
    )[0]
    assert "enableThinking: enableThinkingOverride ?? null" in result_block


def test_release_regression_manifest_real_ui_script_rejects_reasoning_only_visible_turns():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    assert "visibleAssistantAfterEachUser" in source
    assert "UI turn ended with empty visible assistant content" in source
    assert "requestedEnableThinking === true && !visibleAssistantTurnsComplete" not in source
    assert "visibleAssistantTurnsComplete" in source


def test_release_regression_manifest_real_ui_script_rejects_visible_tool_probe_corruption():
    script = Path("panel/scripts/live-real-ui-model-proof.mjs")
    source = script.read_text(encoding="utf-8")

    semantics_block = source.split("function namedToolProbeSemanticsOk(result)", 1)[1].split(
        "function countMatches",
        1,
    )[0]
    assert "visibleToolSemanticsOk" in semantics_block
    assert "real_ui_tool_probe_2.txt" in semantics_block
    assert "REAL_UI_LIVE_TOOL_ONE" in semantics_block
    assert "RE:AL_UI_LIVE_TOOL_TWO" in semantics_block
    assert "strictExactReplyOk" in semantics_block
    assert "reply exactly:" in semantics_block
    assert "&& visibleToolSemanticsOk" in semantics_block
    assert "&& strictExactReplyOk" in semantics_block


def test_release_regression_manifest_current_sweep_rejects_missing_dev_ui_proof(tmp_path):
    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["dev_ui_proof"]["status"] == "fail"
    assert (
        "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-proof.json"
        in result["dev_ui_proof"]["missing"]
    )


def test_release_regression_manifest_current_sweep_rejects_fake_dev_ui_screenshots(tmp_path):
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    screenshot = tmp_path / CURRENT_DEV_UI_PROOF_ARTIFACTS["chat_settings_screenshot"]
    screenshot.write_bytes(b"png-placeholder")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["dev_ui_proof"]["status"] == "fail"
    assert "invalid_png:chat_settings_screenshot" in result["dev_ui_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_wrong_dev_ui_worktree(tmp_path):
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_DEV_UI_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["repoDir"] = "/tmp/wrong-vmlx-worktree"
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["dev_ui_proof"]["status"] == "fail"
    assert "proof_repo_dir_mismatch" in result["dev_ui_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_missing_real_ui_live_model_proof(tmp_path):
    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-current-source-20260531-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )
    assert (
        "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json"
        in result["real_ui_live_model_proof"]["missing"]
    )


def test_release_regression_manifest_real_ui_live_model_rows_include_ling_bailing_slice():
    rows = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS

    assert rows["zaya_text"]["model_name"] == "ZAYA1-8B-MXFP4"
    assert rows["zaya_text_cachecontrols"]["model_name"] == "ZAYA1-8B-MXFP4"
    assert rows["zaya_text_cachecontrols"]["family"] == "zaya_text"
    assert (
        rows["zaya_text_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-proof.json"
    )
    assert rows["zaya_text_responses_tools"]["model_name"] == "ZAYA1-8B-MXFP4"
    assert rows["zaya_text_responses_tools"]["family"] == "zaya_text"
    assert (
        rows["zaya_text_responses_tools"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-tools-cachecontrols-disk-terminal-fix-20260601-proof.json"
    )
    assert rows["zaya_vl_image"]["model_path"] == "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_image"]["model_name"] == "ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_image"]["family"] == "zaya_vl"
    assert (
        rows["zaya_vl_image"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-current-source-20260531-proof.json"
    )
    assert rows["zaya_vl_cachecontrols"]["model_path"] == "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_cachecontrols"]["model_name"] == "ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_cachecontrols"]["family"] == "zaya_vl"
    assert (
        rows["zaya_vl_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-cachecontrols-20260527-proof.json"
    )
    assert rows["zaya_vl_responses_tools_cachecontrols"]["model_path"] == "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_responses_tools_cachecontrols"]["model_name"] == "ZAYA1-VL-8B-JANGTQ4"
    assert rows["zaya_vl_responses_tools_cachecontrols"]["family"] == "zaya_vl"
    assert (
        rows["zaya_vl_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    assert all(
        "point-leak" not in row["proof"]
        for row in rows.values()
    )
    assert all(
        "zaya-vl-responses-tools-cachecontrols-localonly" not in row["proof"]
        for row in rows.values()
    )
    assert rows["ling_bailing_jangtq"]["model_path"] == "/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ"
    assert rows["ling_bailing_jangtq"]["model_name"] == "Ling-2.6-flash-JANGTQ"
    assert (
        rows["ling_bailing_jangtq"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json"
    )
    assert rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["model_path"] == "/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ"
    assert rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["model_name"] == "Ling-2.6-flash-JANGTQ"
    assert rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["family"] == "ling_bailing"
    assert (
        rows["ling_bailing_jangtq_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530-proof.json"
    )
    assert (
        rows["gemma4"]["model_path"]
        == "/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    )
    assert rows["gemma4"]["model_name"] == "Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    assert (
        rows["gemma4"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-stricttools-max768-visible-20260531-proof.json"
    )
    assert rows["gemma4_reasoning"]["model_name"] == "Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    assert rows["gemma4_reasoning"]["family"] == "gemma4"
    assert (
        rows["gemma4_reasoning"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-reasoningonly-max768-visible-20260531-proof.json"
    )
    assert rows["gemma4_cachecontrols"]["model_name"] == "Gemma-4-26B-A4B-it-JANG_4M-CRACK"
    assert rows["gemma4_cachecontrols"]["family"] == "gemma4"
    assert (
        rows["gemma4_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-l2storage-20260531-proof.json"
    )
    assert (
        rows["qwen36_mxfp4_crack"]["model_path"]
        == "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
    )
    assert rows["qwen36_mxfp4_crack"]["model_name"] == "Qwen3.6-27B-JANG_4M-MTP"
    assert rows["qwen36_mxfp4_crack"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json"
    )
    assert (
        rows["qwen36_mxfp4_crack_video"]["model_path"]
        == "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
    )
    assert rows["qwen36_mxfp4_crack_video"]["model_name"] == "Qwen3.6-27B-JANG_4M-MTP"
    assert rows["qwen36_mxfp4_crack_video"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack_video"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max256-20260607-proof.json"
    )
    assert rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["model_path"] == (
        "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
    )
    assert (
        rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["model_name"]
        == "Qwen3.6-27B-JANG_4M-MTP"
    )
    assert rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json"
    )
    assert rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["model_path"] == (
        "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
    )
    assert (
        rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["model_name"]
        == "Qwen3.6-27B-JANG_4M-MTP"
    )
    assert rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["family"] == "qwen36"
    assert (
        rows["qwen36_mxfp4_crack_responses_stricttools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json"
    )
    assert (
        rows["minimax_m27_small_jangtq"]["model_path"]
        == "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert rows["minimax_m27_small_jangtq"]["model_name"] == "MiniMax-M2.7-Small-JANGTQ"
    assert rows["minimax_m27_small_jangtq"]["family"] == "minimax"
    assert (
        rows["minimax_m27_small_jangtq"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-jangtq-20260527-proof.json"
    )
    assert rows["minimax_m27_small_responses_tools_cachecontrols"]["model_path"] == (
        "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_responses_tools_cachecontrols"]["model_name"]
        == "MiniMax-M2.7-Small-JANGTQ"
    )
    assert rows["minimax_m27_small_responses_tools_cachecontrols"]["family"] == "minimax"
    assert (
        rows["minimax_m27_small_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-proof.json"
    )
    assert rows["minimax_m27_small_responses_stricttools_cachecontrols"]["model_path"] == (
        "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_responses_stricttools_cachecontrols"]["model_name"]
        == "MiniMax-M2.7-Small-JANGTQ"
    )
    assert rows["minimax_m27_small_responses_stricttools_cachecontrols"]["family"] == "minimax"
    assert (
        rows["minimax_m27_small_responses_stricttools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    assert rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["model_path"] == (
        "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["model_name"]
        == "MiniMax-M2.7-Small-JANGTQ"
    )
    assert (
        rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["family"]
        == "minimax"
    )
    assert (
        rows["minimax_m27_small_installed_responses_tools_cachecontrols_singleid"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-installed-responses-tools-cachecontrols-singleid-20260528-proof.json"
    )
    assert "minimax_m27_jangtq_k_issue179" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed_hi_auto" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed_hi_thinking" not in rows
    assert "minimax_m27_jangtq_k_issue179_installed_512" not in rows
    manifest_rows = {row["id"]: row for row in build_manifest()["rows"]}
    manifest_row = manifest_rows["electron-dev-ui-real-model-live-slice"]
    joined = "\n".join(
        manifest_row["proves"] + manifest_row["commands"] + manifest_row["artifacts"]
    )
    assert CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT in joined
    assert "run_issue175_179_release_boundary_audit.py" in joined
    assert "GitHub issues #175-#179" in joined
    assert CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT in joined
    assert "run_issue179_minimax_k_root_cause_audit.py" in joined
    assert "run_issue179_responses_cancel_probe.py" in joined
    assert "Responses cancel-404 boundary" in joined
    assert CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT in joined
    assert "run_installed_app_runtime_parity_audit.py" in joined
    assert "installed /Applications/vMLX.app runtime parity audit" in joined
    assert CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT in joined
    assert "run_issue175_177_installed_runtime_audit.py" in joined
    assert "installed-app runtime parity on issues #175-#177" in joined
    assert CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT in joined
    assert "run_issue175_177_live_runtime_audit.py" in joined
    assert "installed-app live runtime audit on issues #175-#177" in joined
    assert (
        rows["nemotron_omni_nano_jangtq"]["model_path"]
        == "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert rows["nemotron_omni_nano_jangtq"]["model_name"] == "Nemotron-Omni-Nano-JANGTQ-CRACK"
    assert rows["nemotron_omni_nano_jangtq"]["family"] == "nemotron_omni"
    assert (
        rows["nemotron_omni_nano_jangtq"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527-proof.json"
    )
    assert rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["model_path"] == (
        "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert (
        rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["model_name"]
        == "Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["family"] == "nemotron_omni"
    assert (
        rows["nemotron_omni_nano_responses_tools_reasoning_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601-proof.json"
    )
    assert rows["nemotron_omni_nano_responses_reasoning"]["model_path"] == (
        "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert (
        rows["nemotron_omni_nano_responses_reasoning"]["model_name"]
        == "Nemotron-Omni-Nano-JANGTQ-CRACK"
    )
    assert rows["nemotron_omni_nano_responses_reasoning"]["family"] == "nemotron_omni"
    assert (
        rows["nemotron_omni_nano_responses_reasoning"]["proof"]
        == "docs/internal/agent-notes/diagnostic-real-ui-live-model-nemotron-omni-nano-responses-reasoning-20260531-proof.json"
    )
    assert rows["hy3_jangtq2"]["model_path"] == "/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2"
    assert rows["hy3_jangtq2"]["model_name"] == "Hy3-preview-JANGTQ2"
    assert rows["hy3_jangtq2"]["family"] == "hy3"
    assert (
        rows["hy3_jangtq2"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-proof.json"
    )
    assert rows["hy3_jangtq2_responses_tools_cachecontrols"]["model_path"] == (
        "/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2"
    )
    assert rows["hy3_jangtq2_responses_tools_cachecontrols"]["model_name"] == "Hy3-preview-JANGTQ2"
    assert rows["hy3_jangtq2_responses_tools_cachecontrols"]["family"] == "hy3"
    assert (
        rows["hy3_jangtq2_responses_tools_cachecontrols"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530-proof.json"
    )
    assert rows["step37_flash_jang2l"]["model_path"] == (
        "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
    )
    assert rows["step37_flash_jang2l"]["model_name"] == "Step-3.7-Flash-JANG_2L"
    assert rows["step37_flash_jang2l"]["family"] == "step37"
    assert (
        rows["step37_flash_jang2l"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-devbuild-tools-image-cache-templateclean-20260531-proof.json"
    )
    assert rows["step37_flash_jang2l_reasoning"]["model_path"] == (
        "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
    )
    assert rows["step37_flash_jang2l_reasoning"]["model_name"] == "Step-3.7-Flash-JANG_2L"
    assert rows["step37_flash_jang2l_reasoning"]["family"] == "step37"
    assert (
        rows["step37_flash_jang2l_reasoning"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-reasoning-max768-20260531-proof.json"
    )
    assert rows["step37_flash_jang2l_l2storage"]["model_path"] == (
        "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
    )
    assert rows["step37_flash_jang2l_l2storage"]["model_name"] == "Step-3.7-Flash-JANG_2L"
    assert rows["step37_flash_jang2l_l2storage"]["family"] == "step37"
    assert (
        rows["step37_flash_jang2l_l2storage"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-cachecontrols-l2storage-20260531-proof.json"
    )
    assert rows["step37_flash_jang2l_tool_l2storage"]["model_path"] == (
        "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L"
    )
    assert rows["step37_flash_jang2l_tool_l2storage"]["model_name"] == "Step-3.7-Flash-JANG_2L"
    assert rows["step37_flash_jang2l_tool_l2storage"]["family"] == "step37"
    assert (
        rows["step37_flash_jang2l_tool_l2storage"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-tools-l2storage-pagedlocked-after-subtype-ui-fix-20260531-proof.json"
    )
    assert rows["lfm25_moe_a1b"]["model_path"] == (
        "/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP4"
    )
    assert rows["lfm25_moe_a1b"]["model_name"] == "LFM2.5-8B-A1B-MXFP4"
    assert rows["lfm25_moe_a1b"]["family"] == "lfm25"
    assert (
        rows["lfm25_moe_a1b"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json"
    )
    assert (
        rows["lfm25_moe_a1b_responses"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json"
    )
    assert (
        rows["lfm25_moe_a1b_responses_delta"]["proof"]
        == "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json"
    )


def test_release_regression_manifest_real_ui_requires_step37_and_lfm25():
    assert "step37" in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES
    assert "lfm25" in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES

    assert REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["step37"] == (
        *tuple(
            surface
            for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
            if surface != "video_where_supported"
        ),
        "tool_l2_cache_integrated",
        "vl_tool_l2_cache_integrated",
    )
    assert REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["lfm25"] == tuple(
        surface
        for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
        if surface
        not in {
            "reasoning_display",
            "vl_image",
            "video_where_supported",
        }
    ) + ("tool_l2_cache_integrated",)


def test_release_regression_manifest_real_ui_artifact_inventory_covers_every_row():
    artifacts = set(CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS.values())

    for row_id, row in CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.items():
        assert row["proof"] in artifacts, row_id
        assert row["chat_screenshot"] in artifacts, row_id


def test_release_regression_manifest_current_sweep_rejects_fake_real_ui_live_model_proof(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["server"]["health"] = {"status": "ok", "model_loaded": False}
    proof["chat"]["rawParserTagLeak"] = True
    proof["chat"]["cjkLeakCount"] = 3
    proof["cache"]["cacheHitTokens"] = 0
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "server_health_not_real_loaded_model" in result["real_ui_live_model_proof"]["failures"]
    assert "raw_parser_or_reasoning_tag_leak" in result["real_ui_live_model_proof"]["failures"]
    assert "visible_language_leak" in result["real_ui_live_model_proof"]["failures"]
    assert "cache_hit_tokens_missing" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_real_ui_proof_requires_visible_multi_turn_assistant_text(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["chat"]["turns"] = [
        {"role": "user", "content": "Say READY."},
        {"role": "assistant", "content": "READY"},
        {"role": "user", "content": "Say READY again."},
        {"role": "assistant", "content": ""},
    ]
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "visible_multi_turn_chat_not_proven" in result[
        "real_ui_live_model_proof"
    ]["failures"]


def test_release_regression_manifest_real_ui_proof_rejects_zaya_point_tag_leak(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["chat"]["rawParserTagLeak"] = False
    proof["chat"]["turns"][1]["content"] = "<|point_start|>function leaked into visible UI"
    proof["chat"]["finalVisibleText"] = "<|point_start|>function leaked into visible UI"
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "raw_parser_or_reasoning_tag_leak" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_real_ui_proof_rejects_reasoning_numeric_garbage(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["chat"]["reasoningText"] = (
        "1 2 3 4 5 6 7 8 9 10 11 12 [Generation interrupted]"
    )
    proof["chat"].pop("reasoningNumericRunCount", None)
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "reasoning_language_or_numeric_leak" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_failed_real_ui_live_model_proof(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["status"] = "fail"
    proof["failureStage"] = "renderer_real_ui_chat_send_message"
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "proof_status_failed" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_real_ui_proof_without_request_contract(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("requestContract", None)
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert "request_contract_missing:zaya_text" in result["real_ui_live_model_proof"]["failures"]


def test_release_regression_manifest_current_sweep_rejects_real_ui_request_contract_mismatch(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = tmp_path / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["proof"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.setdefault("chatOverrides", {})["maxTokens"] = 64
    proof["requestContract"]["requestMaxTokens"] = 512
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert (
        "request_contract_mismatch:zaya_text:maxTokens"
        in result["real_ui_live_model_proof"]["failures"]
    )


def test_release_regression_manifest_current_sweep_requires_lfm_step_extensive_tool_churn(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["lfm25_moe_a1b_responses_proof"]
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.setdefault("eventCounts", {})["tool"] = 3
    proof["persistedToolCount"] = 3
    proof["persistedToolsByMessage"] = [
        [
            {"phase": "result", "toolName": "run_command"},
        ],
        [
            {"phase": "result", "toolName": "run_command"},
        ],
    ]
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert (
        "extensive_tool_churn_missing:lfm25_moe_a1b_responses"
        in result["real_ui_live_model_proof"]["failures"]
    )


def test_release_regression_manifest_current_sweep_rejects_result_only_extensive_tool_churn(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["lfm25_moe_a1b_responses_proof"]
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.setdefault("eventCounts", {})["tool"] = 24
    proof["persistedToolCount"] = 24
    proof["persistedToolsByMessage"] = [
        [{"phase": "result", "toolName": "run_command"} for _ in range(12)],
        [{"phase": "result", "toolName": "run_command"} for _ in range(12)],
    ]
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "fail"
    assert (
        "extensive_tool_churn_missing:lfm25_moe_a1b_responses"
        in result["real_ui_live_model_proof"]["failures"]
    )


def test_release_regression_manifest_accepts_structured_electron_dev_launch_when_log_tail_rotates(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    proof_path = (
        tmp_path
        / CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS["lfm25_moe_a1b_proof"]
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["appLogTail"] = [
        "[CHAT] Tool execution iteration 1 (1 tool calls)",
        "[CHAT] Builtin tool: run_command",
        "[CHAT] Tool loop completed after 1 iteration(s)",
    ]
    proof["uiLaunchMode"] = "electron-dev"
    proof["uiCommand"] = [
        "npm",
        "run",
        "dev",
        "--",
        "--",
        "--user-data-dir=/tmp/vmlx-real-ui-userdata",
        "--remote-debugging-port=55895",
    ]
    proof_path.write_text(json.dumps(proof) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_live_model_proof"]["status"] == "pass"
    assert "electron_dev_renderer_url_missing" not in result[
        "real_ui_live_model_proof"
    ]["failures"]
    assert "electron_dev_launch_log_missing" not in result[
        "real_ui_live_model_proof"
    ]["failures"]


def test_release_regression_manifest_real_ui_matrix_tracks_mixed_minimax_identity(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)
    matrix = result["real_ui_live_model_matrix"]

    assert matrix["status"] == "open"
    assert matrix["release_blocker"] == "Real Electron UI cross-family live model matrix is release-cleared"
    assert matrix["covered_families"]["zaya_text"]["status"] == "pass"
    assert "chat_completions" in matrix["covered_families"]["zaya_text"]["covered_surfaces"]
    assert "cache_hit_telemetry" in matrix["covered_families"]["zaya_text"]["covered_surfaces"]
    assert matrix["covered_families"]["ling_bailing"]["status"] == "pass"
    assert "responses_api" not in matrix["missing_surfaces"]
    assert "responses_delta_streaming" not in matrix["missing_surfaces"]
    assert "long_tool_loop" not in matrix["missing_surfaces"]
    assert "server_cache_controls" not in matrix["missing_surfaces"]
    assert "vl_image" not in matrix["missing_surfaces"]
    assert matrix["covered_families"]["zaya_vl"]["status"] == "pass"
    assert "vl_image" in matrix["covered_families"]["zaya_vl"]["covered_surfaces"]
    assert "gemma4" in matrix["covered_families"]
    assert matrix["covered_families"]["gemma4"]["status"] == "pass"
    assert "qwen36" not in matrix["missing_families"]
    assert matrix["covered_families"]["qwen36"]["status"] == "pass"
    assert matrix["covered_families"]["qwen36"]["allowed_model_variant_union"] is True
    assert matrix["covered_families"]["qwen36"]["mixed_model_identity_union"] is False
    assert matrix["covered_families"]["qwen36"]["missing_required_model_variants"] == []
    assert "cache_hit_telemetry" in matrix["covered_families"]["qwen36"]["covered_surfaces"]
    assert matrix["covered_families"]["minimax"]["status"] == "pass"
    assert matrix["covered_families"]["minimax"]["modelNames"] == [
        "MiniMax-M2.7-Small-JANGTQ",
    ]
    assert matrix["covered_families"]["minimax"]["mixed_model_identity_union"] is False
    assert matrix["mixed_model_identity_families"] == []
    assert matrix["covered_families"]["nemotron_omni"]["status"] == "pass"
    assert matrix["covered_families"]["hy3"]["status"] == "pass"
    assert matrix["missing_families"] == ["dsv4"]
    assert matrix["resource_blockers"]["dsv4"]["reason"] == "insufficient_memory"
    assert matrix["unblocked_non_mimo_status"] == "pass"
    assert matrix["unblocked_non_mimo_missing_families"] == []
    assert matrix["unblocked_non_mimo_partial_families"] == []
    assert matrix["unblocked_non_mimo_excluded_families"] == ["dsv4"]


def test_release_regression_manifest_current_sweep_requires_dsv4_memory_preflight_when_real_ui_missing(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_dsv4_memory_preflight"]["status"] == "missing"
    assert (
        result["real_ui_dsv4_memory_preflight"]["artifact"]
        == CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    )
    assert result["status"] == "fail"


def test_release_regression_manifest_real_ui_matrix_records_dsv4_memory_blocker(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "generated_at": "2026-06-01T17:56:03.986322+00:00",
                "model_path": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 80.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "required_available_gb": 120.0,
                "required_free_gb": 120.0,
                "min_free_gb": 120.0,
                "memory_gap_gb": 80.85,
                "available_for_gate_gb": 39.15,
                "psutil_available_gb": 104.0,
                "psutil_memory_gap_gb": 16.0,
                "memory_pressure_free_percent": 95,
                "inactive_file_cache_gb": 30.0,
                "launch_decision": "do_not_launch",
                "launch_allowed": False,
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["real_ui_dsv4_memory_preflight"]["status"] == "pass"
    assert result["real_ui_live_model_matrix"]["resource_blockers"]["dsv4"] == {
        "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
        "reason": "insufficient_memory",
        "model_path": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
        "model_size_gb": 80.0,
        "required_available_gb": 120.0,
        "required_free_gb": 120.0,
        "min_free_gb": 120.0,
        "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
        "free_plus_speculative_purgeable_gb": 39.15,
        "available_for_gate_gb": 39.15,
        "memory_gap_gb": 80.85,
        "psutil_available_gb": 104.0,
        "psutil_memory_gap_gb": 16.0,
        "memory_pressure_free_percent": 95,
        "inactive_file_cache_gb": 30.0,
        "did_not_launch": True,
        "launch_decision": "do_not_launch",
        "launch_allowed": False,
        "launch_blockers": ["insufficient_memory"],
        "active_heavy_process_count": 0,
        "active_heavy_processes": [],
        "top_memory_processes": [
            {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
        ],
    }
    assert result["real_ui_live_model_matrix"]["unblocked_non_mimo_status"] == "pass"
    assert (
        result["real_ui_live_model_matrix"][
            "unblocked_non_mimo_missing_families"
        ]
        == []
    )
    assert (
        result["real_ui_live_model_matrix"][
            "unblocked_non_mimo_partial_families"
        ]
        == []
    )
    assert result["real_ui_live_model_matrix"][
        "unblocked_non_mimo_excluded_families"
    ] == ["dsv4"]


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_fails_on_missing_family(
):
    matrix = {
        "missing_families": ["mimo_v2", "dsv4", "qwen36"],
        "partial_families": [],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "open"
    assert matrix["unblocked_non_mimo_missing_families"] == ["mimo_v2", "qwen36"]
    assert matrix["unblocked_non_mimo_partial_families"] == []


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_passes_when_only_mimo_and_resource_blocked_dsv4_are_open():
    matrix = {
        "missing_families": ["mimo_v2", "dsv4"],
        "partial_families": [],
        "mixed_model_identity_families": [],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "open"
    assert matrix["unblocked_non_mimo_missing_families"] == ["mimo_v2"]
    assert matrix["unblocked_non_mimo_partial_families"] == []
    assert matrix["unblocked_non_mimo_excluded_families"] == ["dsv4"]


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_excludes_runtime_blocked_step37():
    matrix = {
        "missing_families": ["mimo_v2", "dsv4", "step37"],
        "partial_families": ["step37"],
        "mixed_model_identity_families": [],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
        "runtime_blockers": {
            "step37": {"reason": "missing_mlx_vlm_step3p7_vlm_runtime"}
        },
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "open"
    assert matrix["unblocked_non_mimo_missing_families"] == ["mimo_v2"]
    assert matrix["unblocked_non_mimo_partial_families"] == []
    assert matrix["unblocked_non_mimo_excluded_families"] == [
        "dsv4",
        "step37",
    ]


def test_release_regression_manifest_real_ui_unblocked_non_mimo_status_rejects_mixed_identity():
    matrix = {
        "missing_families": ["mimo_v2", "dsv4"],
        "partial_families": [],
        "mixed_model_identity_families": ["minimax"],
        "resource_blockers": {"dsv4": {"reason": "insufficient_memory"}},
    }

    _annotate_real_ui_unblocked_non_mimo_status(matrix)

    assert matrix["unblocked_non_mimo_status"] == "open"
    assert matrix["unblocked_non_mimo_missing_families"] == ["mimo_v2"]
    assert matrix["unblocked_non_mimo_partial_families"] == ["minimax"]


def test_release_regression_manifest_rejects_dsv4_preflight_under_model_margin(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "model_path": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 100.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "required_available_gb": 120.0,
                "memory_gap_gb": 80.85,
                "launch_decision": "do_not_launch",
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [],
                "active_heavy_process_count": 0,
                "launch_blockers": ["insufficient_memory"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "required_available_gb_missing_model_margin" in result["failures"]


def test_release_regression_manifest_rejects_dsv4_preflight_with_active_heavy_process(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_memory",
                "model_path": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                "model_size_gb": 80.0,
                "free_plus_speculative_purgeable_gb": 39.15,
                "required_available_gb": 120.0,
                "memory_gap_gb": 80.85,
                "launch_decision": "do_not_launch",
                "top_memory_processes": [
                    {"pid": 1001, "rss_gb": 10.0, "command": "vMLX"}
                ],
                "active_heavy_processes": [
                    {
                        "pid": 2002,
                        "command": "python tests/cross_matrix/run_runtime_memory_stress_probe.py",
                    }
                ],
                "active_heavy_process_count": 1,
                "launch_blockers": ["insufficient_memory", "active_heavy_process"],
                "did_not_launch": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "active_heavy_processes_not_clear" in result["failures"]


def test_release_regression_manifest_rejects_dsv4_preflight_with_nonzero_active_process_count(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload["active_heavy_process_count"] = 1
    payload["active_heavy_processes"] = []
    preflight_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "active_heavy_process_count_not_zero" in result["failures"]


def test_release_regression_manifest_rejects_dsv4_preflight_missing_generated_at(tmp_path):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload.pop("generated_at", None)
    preflight_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "generated_at_missing" in result["failures"]


def test_release_regression_manifest_rejects_dsv4_preflight_without_vm_stat_memory_source(
    tmp_path,
):
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)
    preflight_path = tmp_path / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload["preflight_memory_source"] = "psutil_available"
    preflight_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)[
        "real_ui_dsv4_memory_preflight"
    ]

    assert result["status"] == "fail"
    assert "preflight_memory_source_not_vm_stat" in result["failures"]


def test_release_regression_manifest_real_ui_matrix_requires_every_family_surface():
    native_cache = {
        "family": "plain_kv",
        "schema": "plain_kv_v1",
        "cache_type": "paged_kv",
        "components": ["attention_kv"],
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }
    cache_stats = {
        "before": {"scheduler_cache": {"hits": 0}},
        "after": {
            "scheduler_cache": {"hits": 1},
            "block_disk_cache": {"disk_hits": 1},
            "cache_totals": {"l2_tokens_on_disk": 12},
        },
        "cacheHitTokens": 12,
    }
    lfm_native_cache = {
        "family": "lfm2_moe",
        "schema": "hybrid_ssm_v1",
        "cache_type": "hybrid_ssm_typed",
        "components": ["attention_kv", "ssm_companion_state", "async_rederive"],
        "generic_turboquant_kv": {
            "enabled": False,
            "reason": "hybrid_ssm_state",
        },
        "attention_kv_storage_quantization": {
            "enabled": True,
            "mode": "storage_boundary",
            "bits": 4,
            "group_size": 64,
            "applies_to": "attention_kv_layers_only",
            "ssm_policy": "native_companion_state",
        },
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }
    step_native_cache = {
        "family": "mixed_attention",
        "schema": "mixed_swa_kv_v1",
        "cache_type": "mixed_swa_kv",
        "components": [
            "full_attention_kv",
            "sliding_window_kv",
            "rotating_window_metadata",
        ],
        "generic_turboquant_kv": {"enabled": False, "reason": "not_active"},
        "storage_quantization": {
            "enabled": True,
            "mode": "storage_boundary",
            "bits": 4,
            "group_size": 64,
            "applies_to": "full_attention_kv_only",
            "metadata_policy": "preserve_rotating_window_metadata",
        },
        "prefix": True,
        "paged": True,
        "block_disk_l2": True,
    }
    proofs = {
        family: {
            "modelName": f"{family}-model",
            "appLogTail": ["start electron app"],
            "server": {
                "health": {
                    "status": "healthy",
                    "model_loaded": True,
                    "native_cache": native_cache,
                    "kv_cache_quantization": {
                        "enabled": True,
                        "bits": 4,
                        "group_size": 64,
                    },
                    "scheduler": {
                        "last_cache_execution": {
                            "cache_detail": "paged",
                            "cached_tokens": 12,
                        }
                    },
                }
            },
            "chat": {
                "turns": [
                    {"role": "user", "content": "Say READY."},
                    {"role": "assistant", "content": "READY"},
                    {"role": "user", "content": "Repeat READY."},
                    {"role": "assistant", "content": "READY"},
                ],
                "rawParserTagLeak": False,
                "cjkLeakCount": 0,
                "koreanLeakCount": 0,
            },
            "cache": cache_stats,
            "rendererWireApi": "responses",
            "eventCounts": {
                "complete": 2,
                "stream": 4,
                "tool": 24,
                "reasoningDone": 1,
            },
            "streamTrace": [
                {
                    "messageId": f"{family}-stream",
                    "count": 4,
                    "firstFullContent": "o",
                    "lastFullContent": "ok streamed",
                },
                {
                    "messageId": f"{family}-stream-second",
                    "count": 4,
                    "firstFullContent": "s",
                    "lastFullContent": "second ok streamed",
                },
            ],
            "persistedToolCount": 24,
            "persistedToolsByMessage": [
                [
                    {"phase": "calling", "toolName": "run_command"},
                    {"phase": "executing", "toolName": "run_command"},
                    {"phase": "result", "toolName": "run_command"},
                    *(
                        {"phase": "result", "toolName": "run_command"}
                        for _ in range(7)
                    ),
                ],
                [
                    {"phase": "calling", "toolName": "write_file"},
                    {"phase": "executing", "toolName": "write_file"},
                    {"phase": "result", "toolName": "write_file"},
                    *(
                        {"phase": "result", "toolName": "write_file"}
                        for _ in range(7)
                    ),
                ],
            ],
            "persistedReasoningCount": 1,
            "requestedBuiltinTools": True,
            "chatOverrides": {"builtinToolsEnabled": True},
            "serverLogTail": [
                "INFO:vmlx_engine.server:Resolved sampling kwargs "
                "route=/v1/responses model=family-model "
                "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
            ],
            "serverCacheControls": {"verified": True},
            "media": {"imageVerified": True, "videoVerified": True},
        }
        for family in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES
    }
    proofs["qwen36"]["modelName"] = "Qwen3.6-27B-JANG_4M-MTP"
    proofs["qwen36"]["modelPath"] = "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP"
    proofs["qwen36_35b_mxfp8_mtp_responses_tools_cachecontrols"] = json.loads(
        json.dumps(proofs["qwen36"])
    )
    proofs["qwen36_35b_mxfp8_mtp_responses_tools_cachecontrols"][
        "modelName"
    ] = "Qwen3.6-35B-A3B-MXFP8-MTP"
    proofs["qwen36_35b_mxfp8_mtp_responses_tools_cachecontrols"][
        "modelPath"
    ] = "/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP"
    proofs["lfm25"]["server"]["health"]["native_cache"] = lfm_native_cache
    proofs["lfm25"]["cache"]["after"]["ssm_companion"] = {
        "disk": {
            "enabled": True,
            "entries": 1,
            "total_tokens_on_disk": 64,
            "total_cached_tokens": 64,
            "stores": 1,
        }
    }
    proofs["lfm25"]["cache"]["after"]["cache_totals"].update(
        {
            "l2_ssm_tokens_on_disk": 64,
            "ssm_tokens_on_disk": 64,
        }
    )
    proofs["lfm25"]["appLogTail"] = [
        "start electron app",
        "[CHAT] Response complete: 134 tokens in 1.1s "
        "(356.4 t/s, live=228.3 t/s, TTFT: 0.07s, "
        "pp: 725 tokens (618 cached), 10984.8 pp/s, usage=server)",
        "[CHAT] Response complete: 136 tokens in 1.1s "
        "(353.2 t/s, live=228.3 t/s, TTFT: 0.07s, "
        "pp: 1024 tokens (917 cached), 14027.4 pp/s, usage=server)",
    ]
    proofs["step37"]["server"]["health"]["native_cache"] = step_native_cache
    proofs["step37"]["appLogTail"] = [
        "start electron app",
        "[CHAT] Response complete: 78 tokens in 11.1s "
        "(18.9 t/s, live=50.2 t/s, TTFT: 0.43s, "
        "pp: 2529 tokens (2441 cached), 5813.8 pp/s, usage=server)",
        "[CHAT] Response complete: 86 tokens in 7.7s "
        "(19.5 t/s, live=49.9 t/s, TTFT: 0.44s, "
        "pp: 2700 tokens (2612 cached), 6178.5 pp/s, usage=server)",
    ]
    proofs["gemma4"]["persistedToolCount"] = 0
    proofs["gemma4"]["persistedToolsByMessage"] = []

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": proofs}
    )

    assert matrix["missing_families"] == []
    assert matrix["missing_surfaces"] == []
    assert matrix["status"] == "open"
    assert matrix["partial_families"] == ["gemma4"]
    assert "long_tool_loop" in matrix["covered_families"]["gemma4"]["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_uses_family_specific_media_requirements():
    def complete_proof(*, image=False, video=False):
        return {
            "modelName": "architecture-aware-model",
            "appLogTail": ["start electron app"],
            "server": {
                "health": {
                    "status": "healthy",
                    "model_loaded": True,
                    "native_cache": {
                        "family": "plain_kv",
                        "schema": "plain_kv_v1",
                        "cache_type": "paged_kv",
                        "components": ["attention_kv"],
                        "prefix": True,
                        "paged": True,
                        "block_disk_l2": True,
                    },
                }
            },
            "chat": {
                "turns": [
                    {"role": "user", "content": "Say READY."},
                    {"role": "assistant", "content": "READY"},
                    {"role": "user", "content": "Repeat READY."},
                    {"role": "assistant", "content": "READY"},
                ],
                "rawParserTagLeak": False,
                "cjkLeakCount": 0,
                "koreanLeakCount": 0,
            },
            "cache": {
                "before": {"scheduler_cache": {"hits": 0}},
                "after": {
                    "scheduler_cache": {"hits": 1},
                    "block_disk_cache": {"disk_hits": 1},
                    "cache_totals": {"l2_tokens_on_disk": 12},
                },
                "cacheHitTokens": 12,
            },
            "rendererWireApi": "responses",
            "eventCounts": {
                "complete": 2,
                "stream": 4,
                "tool": 3,
                "reasoningDone": 1,
            },
            "streamTrace": [
                {
                    "messageId": "media-proof-stream",
                    "count": 4,
                    "firstFullContent": "o",
                    "lastFullContent": "ok streamed",
                },
                {
                    "messageId": "media-proof-stream-second",
                    "count": 4,
                    "firstFullContent": "s",
                    "lastFullContent": "second ok streamed",
                },
            ],
            "persistedToolCount": 6,
            "persistedToolsByMessage": [
                [
                    {"phase": "calling", "toolName": "run_command"},
                    {"phase": "executing", "toolName": "run_command"},
                    {"phase": "result", "toolName": "run_command"},
                ],
                [
                    {"phase": "calling", "toolName": "write_file"},
                    {"phase": "executing", "toolName": "write_file"},
                    {"phase": "result", "toolName": "write_file"},
                ],
            ],
            "persistedReasoningCount": 1,
            "requestedBuiltinTools": True,
            "chatOverrides": {"builtinToolsEnabled": True},
            "serverLogTail": [
                "INFO:vmlx_engine.server:Resolved sampling kwargs "
                "route=/v1/responses model=architecture-aware-model "
                "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
            ],
            "serverCacheControls": {"verified": True},
            "media": {"imageVerified": image, "videoVerified": video},
        }

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "zaya_text": complete_proof(),
                "zaya_vl_image": complete_proof(image=True),
                "qwen36_mxfp4_crack": complete_proof(image=True),
            },
        }
    )

    assert "vl_image" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["zaya_text"]
    assert "video_where_supported" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["zaya_text"]
    assert "reasoning_display" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["zaya_text"]
    assert "reasoning_display" not in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["ling_bailing"]
    assert matrix["covered_families"]["zaya_text"]["status"] == "pass"
    assert "vl_image" not in matrix["covered_families"]["zaya_text"]["missing_surfaces"]
    assert matrix["covered_families"]["zaya_vl"]["status"] == "pass"
    assert "video_where_supported" not in matrix["covered_families"]["zaya_vl"]["missing_surfaces"]
    assert matrix["covered_families"]["qwen36"]["status"] == "partial"
    assert "video_where_supported" in matrix["covered_families"]["qwen36"]["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_multi_turn_chat():
    proof = {
        "modelName": "ZAYA1-8B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "zaya",
                    "schema": "zaya_cca_v1",
                    "cache_type": "typed_cca",
                    "components": ["standard_kv", "cca_conv_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "READY"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "streamTrace": [
            {
                "messageId": "stream-one",
                "count": 4,
                "firstFullContent": "R",
                "lastFullContent": "READY one",
            },
            {
                "messageId": "stream-two",
                "count": 4,
                "firstFullContent": "O",
                "lastFullContent": "OK two",
            },
        ],
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "REAL_UI_LIVE_TOOL_ONE",
                },
            ],
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "REAL_UI_LIVE_TOOL_TWO",
                },
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=ZAYA1-8B-MXFP4 "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"zaya_text": proof}}
    )

    zaya = matrix["covered_families"]["zaya_text"]
    assert "multi_turn_chat" not in zaya["covered_surfaces"]
    assert "multi_turn_chat" in zaya["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_cache_hit_telemetry_when_reconstruction_falls_back_to_miss():
    proof = {
        "modelName": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "nemotron_h",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3, "reasoningDone": 1},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
        ],
        "persistedReasoningCount": 1,
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.scheduler:Request resp_a: worker-side paged cache reconstruction failed, treating as cache miss"
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"nemotron_omni_nano_responses_tools_reasoning_cachecontrols": proof}}
    )

    family = matrix["covered_families"]["nemotron_omni"]
    assert "cache_hit_telemetry" not in family["covered_surfaces"]
    assert "cache_hit_telemetry" in family["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_endpoint_l2_disk_storage_for_native_l2():
    proof = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "mixed_attention",
                    "schema": "mixed_swa_kv_v1",
                    "cache_type": "mixed_swa_kv",
                    "components": [
                        "full_attention_kv",
                        "sliding_window_kv",
                        "rotating_window_metadata",
                    ],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {
                    "blocks_on_disk": 0,
                    "total_tokens_on_disk": 0,
                    "disk_hits": 0,
                    "disk_writes": 0,
                },
                "cache_totals": {"l2_tokens_on_disk": 0},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3, "reasoningDone": 1},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
        ],
        "persistedReasoningCount": 1,
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "media": {"imageVerified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    assert "l2_disk_storage" in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["step37"]
    assert "l2_disk_storage" not in step37["covered_surfaces"]
    assert "l2_disk_storage" in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_integrated_tool_l2_cache_proof():
    base = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "mixed_attention",
                    "schema": "mixed_swa_kv_v1",
                    "cache_type": "mixed_swa_kv",
                    "components": [
                        "full_attention_kv",
                        "sliding_window_kv",
                        "rotating_window_metadata",
                    ],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4},
        "streamTrace": [
            {
                "messageId": "step-stream-1",
                "count": 4,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed",
            },
            {
                "messageId": "step-stream-2",
                "count": 4,
                "firstFullContent": "s",
                "lastFullContent": "second ok streamed",
            },
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=step37 "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "media": {"imageVerified": True},
    }
    tool_only = {
        **base,
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {
                    "blocks_on_disk": 0,
                    "total_tokens_on_disk": 0,
                    "disk_writes": 0,
                },
                "cache_totals": {"l2_tokens_on_disk": 0},
            },
            "cacheHitTokens": 12,
        },
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {"phase": "result", "toolName": "run_command"},
            ],
            [
                {"phase": "calling", "toolName": "write_file"},
                {"phase": "executing", "toolName": "write_file"},
                {"phase": "result", "toolName": "write_file"},
            ],
        ],
    }
    l2_only = {
        **base,
        "eventCounts": {"complete": 2, "stream": 4, "tool": 0},
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {
                    "blocks_on_disk": 2,
                    "total_tokens_on_disk": 128,
                    "disk_writes": 2,
                },
                "cache_totals": {"l2_tokens_on_disk": 128},
            },
            "cacheHitTokens": 12,
        },
        "persistedToolCount": 0,
        "persistedToolsByMessage": [],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "step37_flash_jang2l": tool_only,
                "step37_flash_jang2l_l2storage": l2_only,
            },
        }
    )

    step37 = matrix["covered_families"]["step37"]
    assert "tool_l2_cache_integrated" in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["step37"]
    assert "long_tool_loop" in step37["covered_surfaces"]
    assert "l2_disk_storage" in step37["covered_surfaces"]
    assert "tool_l2_cache_integrated" not in step37["covered_surfaces"]
    assert "tool_l2_cache_integrated" in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_records_surface_artifacts():
    proof = _lfm_integrated_matrix_proof()

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    expected_artifact = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS[
        "lfm25_moe_a1b_responses_delta"
    ]["proof"]
    assert lfm25["surface_artifacts"]["tool_l2_cache_integrated"] == [
        expected_artifact
    ]
    assert lfm25["surface_artifacts"]["architecture_cache_policy"] == [
        expected_artifact
    ]
    assert lfm25["surface_artifacts"]["responses_delta_streaming"] == [
        expected_artifact
    ]


def _step37_integrated_vl_tool_l2_matrix_proof() -> dict[str, object]:
    return {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": [
            "start electron app",
            "[CHAT] Response complete: 78 tokens in 11.1s "
            "(18.9 t/s, live=50.2 t/s, TTFT: 0.43s, pp: 2529 tokens "
            "(2441 cached), 5813.8 pp/s, usage=server)",
            "[CHAT] Response complete: 86 tokens in 7.7s "
            "(19.5 t/s, live=49.9 t/s, TTFT: 0.44s, pp: 2700 tokens "
            "(2612 cached), 6178.5 pp/s, usage=server)",
        ],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "mixed_attention",
                    "schema": "mixed_swa_kv_v1",
                    "cache_type": "mixed_swa_kv",
                    "components": [
                        "full_attention_kv",
                        "sliding_window_kv",
                        "rotating_window_metadata",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "mixed_swa_kv",
                    },
                    "storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "full_attention_kv_only",
                        "metadata_policy": "preserve_rotating_window_metadata",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+mixed_swa",
                        "cached_tokens": 2612,
                    }
                },
            }
        },
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 3},
                "block_disk_cache": {
                    "blocks_on_disk": 24,
                    "disk_hits": 31,
                    "disk_writes": 24,
                },
                "cache_totals": {
                    "l2_tokens_on_disk": 3179,
                    "l2_block_tokens_on_disk": 3179,
                },
            },
            "cacheHitTokens": 7101,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "streamTrace": [
            {
                "messageId": "step-stream-1",
                "count": 2,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed one",
            },
            {
                "messageId": "step-stream-2",
                "count": 2,
                "firstFullContent": "t",
                "lastFullContent": "ok streamed two",
            },
        ],
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "real_ui_tool_probe_1.txt REAL_UI_LIVE_TOOL_ONE",
                },
                *({"phase": "result", "toolName": "run_command"} for _ in range(7)),
            ],
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "real_ui_tool_probe_2.txt REAL_UI_LIVE_TOOL_TWO",
                },
                *({"phase": "result", "toolName": "run_command"} for _ in range(7)),
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=step "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "requestContract": {
            "requestMaxTokens": 512,
            "wireApi": "responses",
            "builtinToolsEnabled": True,
        },
        "media": {"imageVerified": True},
    }


def test_release_regression_manifest_real_ui_matrix_requires_step37_vl_tool_l2_same_artifact():
    proof = _step37_integrated_vl_tool_l2_matrix_proof()
    proof["media"] = {"imageVerified": False}

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    assert "tool_l2_cache_integrated" in step37["covered_surfaces"]
    assert "vl_image" not in step37["covered_surfaces"]
    assert "vl_tool_l2_cache_integrated" not in step37["covered_surfaces"]
    assert "vl_tool_l2_cache_integrated" in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_step37_vl_tool_l2_same_artifact():
    proof = _step37_integrated_vl_tool_l2_matrix_proof()

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    expected_artifact = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS[
        "step37_flash_jang2l"
    ]["proof"]
    assert "vl_tool_l2_cache_integrated" in step37["covered_surfaces"]
    assert step37["surface_artifacts"]["vl_tool_l2_cache_integrated"] == [
        expected_artifact
    ]


def test_release_regression_manifest_real_ui_matrix_requires_step37_mixed_swa_storage_quantization():
    proof = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "mixed_attention",
                    "schema": "mixed_swa_kv_v1",
                    "cache_type": "mixed_swa_kv",
                    "components": [
                        "full_attention_kv",
                        "sliding_window_kv",
                        "rotating_window_metadata",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "mixed_swa_kv",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+mixed_swa",
                        "cached_tokens": 2612,
                    }
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {"cacheHitTokens": 2612},
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "requestContract": {"requestMaxTokens": 512, "wireApi": "responses"},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    assert "architecture_cache_policy" not in step37["covered_surfaces"]
    assert "architecture_cache_policy" in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_step37_family_alias_for_mixed_swa_policy():
    proof = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "server": {
            "health": {
                "native_cache": {
                    "family": "step3p7",
                    "schema": "mixed_swa_kv_v1",
                    "cache_type": "mixed_swa_kv",
                    "cache_subtype": "step3p7_full_sliding_kv",
                    "components": [
                        "full_attention_kv",
                        "sliding_window_kv",
                        "rotating_window_metadata",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "not_active",
                    },
                    "storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "full_attention_kv_only",
                        "metadata_policy": "preserve_rotating_window_metadata",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
            }
        },
        "cache": {
            "cacheHitTokens": 64,
            "after": {"cache_totals": {"l2_tokens_on_disk": 64}},
        },
    }

    assert _real_ui_architecture_cache_policy_ok("step37", proof) is True


def test_release_regression_manifest_real_ui_matrix_rejects_hybrid_ssm_full_prefill_fallback():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [{"phase": "result", "toolName": "run_command"}],
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.scheduler:Request resp_a: hybrid paged MISS — 448 KV tokens cached but no usable SSM companion, full prefill"
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b": proof}}
    )

    family = matrix["covered_families"]["lfm25"]
    assert "cache_hit_telemetry" not in family["covered_surfaces"]
    assert "cache_hit_telemetry" in family["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_hybrid_ssm_hit_after_warmup_miss():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+ssm",
                        "reconstruction_ok": True,
                    }
                },
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 2},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3},
        "persistedToolCount": 1,
        "persistedToolsByMessage": [
            [{"phase": "result", "toolName": "run_command"}],
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.scheduler:Request resp_a: hybrid paged MISS — 448 KV tokens cached but no usable SSM companion, full prefill",
            "INFO:vmlx_engine.scheduler:Request resp_b: hybrid paged HIT — 320 tokens (KV + 18 SSM layers)",
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b": proof}}
    )

    family = matrix["covered_families"]["lfm25"]
    assert "cache_hit_telemetry" in family["covered_surfaces"]


def test_release_regression_manifest_real_ui_named_tool_probe_accepts_split_file_writes():
    proof = {
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt",
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_rejects_visible_file_semantic_contradiction():
    proof = {
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {
                    "role": "assistant",
                    "content": (
                        "REAL_UI_LIVE_TOOL_TWO\n\n"
                        "The file real_ui_tool_probe_2.txt contains the string "
                        '"Real_UI_Live_Tool_One".'
                    ),
                },
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt",
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert not _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_rejects_malformed_visible_tool_sentinel():
    proof = {
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {
                    "role": "assistant",
                    "content": (
                        "The content of real_ui_tool_probe_2.txt is "
                        "REAL_UI_LIVE_TOOL_TWO.\n\nRE:AL_UI_LIVE_TOOL_TWO"
                    ),
                },
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                        "real_ui_tool_probe_1.txt && cat real_ui_tool_probe_1.txt\n\n"
                        "REAL_UI_LIVE_TOOL_ONE"
                    ),
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert not _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_rejects_explanatory_exact_reply():
    proof = {
        "chat": {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once. "
                        "After the tool result, reply exactly: REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The command has been executed as requested.\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert not _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_named_tool_probe_accepts_exact_reply_contract():
    proof = {
        "chat": {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once. "
                        "After the tool result, reply exactly: REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ]
        },
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "$ printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
                }
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_TWO > "
                        "real_ui_tool_probe_2.txt && cat real_ui_tool_probe_2.txt\n\n"
                        "REAL_UI_LIVE_TOOL_TWO"
                    ),
                }
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
    }

    assert _real_ui_named_tool_probe_semantics_ok(proof)


def test_release_regression_manifest_real_ui_matrix_rejects_forged_tool_reasoning_surfaces():
    proof = {
        "modelName": "ZAYA1-8B-MXFP4",
        "provenSurfaces": [
            "responses_api",
            "long_tool_loop",
            "reasoning_display",
        ],
        "rendererWireApi": "chat",
        "eventCounts": {"complete": 2, "tool": 0, "reasoningDone": 0},
        "persistedToolCount": 0,
        "persistedReasoningCount": 0,
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"gemma4": proof}}
    )

    gemma4 = matrix["covered_families"]["gemma4"]
    assert "responses_api" not in gemma4["covered_surfaces"]
    assert "long_tool_loop" not in gemma4["covered_surfaces"]
    assert "reasoning_display" not in gemma4["covered_surfaces"]
    assert "responses_api" in gemma4["missing_surfaces"]
    assert "long_tool_loop" in gemma4["missing_surfaces"]
    assert "reasoning_display" in gemma4["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_reasoning_only_visible_turns():
    proof = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "step_vl",
                    "schema": "mllm_rotating_kv_v1",
                    "cache_type": "mllm_rotating_paged_kv",
                    "components": ["attention_kv"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
            "turns": [
                {"role": "user", "content": "Say FINAL=STEP37_REASONING_OK."},
                {"role": "assistant", "content": "FINAL=STEP37_REASONING_OK"},
                {"role": "user", "content": "Confirm it again."},
                {"role": "assistant", "content": ""},
            ],
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "requestedEnableThinking": True,
        "eventCounts": {"complete": 2, "reasoningDone": 2},
        "persistedReasoningCount": 2,
        "requestedBuiltinTools": False,
        "chatOverrides": {"builtinToolsEnabled": False},
        "serverCacheControls": {"verified": True},
        "media": {"imageVerified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l_reasoning": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    assert "reasoning_display" not in step37["covered_surfaces"]
    assert "reasoning_display" in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_lfm2_raw_tool_markers():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {"model_name": "LFM2.5-8B-A1B-MXFP4"},
            }
        },
        "chat": {
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
            "turns": [
                {
                    "role": "assistant",
                    "content": "<|tool_call_start|>\nvisible leaked tool marker",
                }
            ],
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 2},
        "persistedToolCount": 2,
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "parser_leak_check" not in lfm25["covered_surfaces"]
    assert "parser_leak_check" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_top_level_parser_leak_flags():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {"model_name": "LFM2.5-8B-A1B-MXFP4"},
            }
        },
        "chat": {
            "turns": [
                {
                    "role": "assistant",
                    "content": "tool result handled without visible parser tags",
                }
            ],
        },
        "rawParserLeak": False,
        "reasoningRawParserLeak": False,
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 3},
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "write_file"},
                {"phase": "result", "toolName": "read_file"},
            ]
        ],
        "persistedToolCount": 2,
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "parser_leak_check" in lfm25["covered_surfaces"]
    assert "parser_leak_check" not in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_responses_delta_streaming():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "cache": {"cacheHitTokens": 64},
        "chat": {
            "turns": [
                {"role": "user", "content": "tool turn one"},
                {"role": "assistant", "content": "complete only"},
            ],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "reasoningRawParserLeak": False,
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 1, "stream": 1},
        "streamTrace": [
            {
                "messageId": "m1",
                "count": 1,
                "firstFullContent": "complete only",
                "lastFullContent": "complete only",
            }
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "responses_delta_streaming" not in lfm25["covered_surfaces"]
    assert "responses_delta_streaming" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_two_turn_responses_delta_streaming():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "cache": {"cacheHitTokens": 64},
        "chat": {
            "turns": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first streamed"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second complete"},
            ],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "reasoningRawParserLeak": False,
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4},
        "streamTrace": [
            {
                "messageId": "first-only",
                "count": 4,
                "firstFullContent": "f",
                "lastFullContent": "first streamed",
            }
        ],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "responses_delta_streaming" not in lfm25["covered_surfaces"]
    assert "responses_delta_streaming" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_generation_defaults_proof():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "cache": {"cacheHitTokens": 64},
        "chat": {
            "turns": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hello"},
            ],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "reasoningRawParserLeak": False,
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 1, "stream": 4},
        "streamTrace": [
            {
                "messageId": "m1",
                "count": 4,
                "firstFullContent": "h",
                "lastFullContent": "hello",
            }
        ],
        "chatOverrides": {
            "temperature": None,
            "topP": None,
            "topK": None,
            "minP": None,
            "repeatPenalty": None,
            "maxTokens": 128,
        },
        "serverLogTail": [],
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "generation_defaults_applied" not in lfm25["covered_surfaces"]
    assert "generation_defaults_applied" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_lfm_architecture_cache_policy():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "plain_kv",
                    "schema": "plain_kv_v1",
                    "cache_type": "paged_kv",
                    "components": ["attention_kv"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                    "generic_turboquant_kv": {
                        "enabled": True,
                        "reason": "plain_attention",
                    },
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {
                    "blocks_on_disk": 1,
                    "total_tokens_on_disk": 12,
                    "disk_hits": 1,
                    "disk_writes": 1,
                },
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {
            "complete": 2,
            "stream": 4,
            "tool": 24,
        },
        "streamTrace": [
            {
                "messageId": "lfm-stream",
                "count": 4,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed",
            }
        ],
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=lfm "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "requestContract": {
            "requestMaxTokens": 512,
            "wireApi": "responses",
            "builtinToolsEnabled": True,
        },
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert (
        "architecture_cache_policy"
        in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["lfm25"]
    )
    assert "architecture_cache_policy" not in lfm25["covered_surfaces"]
    assert "architecture_cache_policy" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_lfm2_family_alias_for_hybrid_ssm_policy():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": [
                        "attention_kv",
                        "ssm_companion_state",
                        "async_rederive",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "hybrid_ssm_state",
                    },
                    "attention_kv_storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "attention_kv_layers_only",
                        "ssm_policy": "native_companion_state",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
            }
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {
                    "l2_tokens_on_disk": 32,
                    "l2_ssm_tokens_on_disk": 16,
                    "ssm_tokens_on_disk": 16,
                },
                "ssm_companion": {
                    "entries": 1,
                    "disk_enabled": True,
                    "disk": {
                        "enabled": True,
                        "entries": 1,
                        "total_tokens_on_disk": 16,
                    },
                },
            },
            "cacheHitTokens": 32,
        },
    }

    assert _real_ui_architecture_cache_policy_ok("lfm25", proof) is True


def test_release_regression_manifest_real_ui_matrix_requires_lfm_responses_cache_detail_usage():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": [
                        "attention_kv",
                        "ssm_companion_state",
                        "async_rederive",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "hybrid_ssm_state",
                    },
                    "attention_kv_storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "attention_kv_layers_only",
                        "ssm_policy": "native_companion_state",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {
                    "blocks_on_disk": 1,
                    "total_tokens_on_disk": 12,
                    "disk_hits": 1,
                    "disk_writes": 1,
                },
                "cache_totals": {
                    "l2_tokens_on_disk": 76,
                    "l2_ssm_tokens_on_disk": 64,
                    "ssm_tokens_on_disk": 64,
                },
                "ssm_companion": {
                    "disk": {
                        "enabled": True,
                        "entries": 1,
                        "total_tokens_on_disk": 64,
                        "total_cached_tokens": 64,
                        "stores": 1,
                    }
                },
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {
            "complete": 2,
            "stream": 4,
            "tool": 24,
        },
        "streamTrace": [
            {
                "messageId": "lfm-stream",
                "count": 4,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed",
            }
        ],
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
            [
                {"phase": "result", "toolName": "run_command"},
                {"phase": "result", "toolName": "write_file"},
            ],
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=lfm "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "requestContract": {
            "requestMaxTokens": 512,
            "wireApi": "responses",
            "builtinToolsEnabled": True,
        },
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert (
        "responses_cache_detail_usage"
        in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["lfm25"]
    )
    assert "responses_cache_detail_usage" not in lfm25["covered_surfaces"]
    assert "responses_cache_detail_usage" in lfm25["missing_surfaces"]

    proof["streamTrace"][0]["lastMetrics"] = {
        "cachedTokens": 917,
        "cacheDetail": "paged+ssm",
    }
    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )
    lfm25 = matrix["covered_families"]["lfm25"]
    assert "responses_cache_detail_usage" in lfm25["covered_surfaces"]
    assert "responses_cache_detail_usage" not in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_integrated_tool_l2_requires_responses_cache_detail():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": [
            "start electron app",
            "[CHAT] Response complete: 134 tokens in 1.1s "
            "(356.4 t/s, live=228.3 t/s, TTFT: 0.07s, pp: 725 tokens "
            "(618 cached), 10984.8 pp/s, usage=server)",
            "[CHAT] Response complete: 136 tokens in 1.1s "
            "(353.2 t/s, live=228.3 t/s, TTFT: 0.07s, pp: 1024 tokens "
            "(917 cached), 14027.4 pp/s, usage=server)",
        ],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": [
                        "attention_kv",
                        "ssm_companion_state",
                        "async_rederive",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "hybrid_ssm_state",
                    },
                    "attention_kv_storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "attention_kv_layers_only",
                        "ssm_policy": "native_companion_state",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
            }
        },
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 3},
                "block_disk_cache": {
                    "blocks_on_disk": 24,
                    "total_tokens_on_disk": 1420,
                    "disk_hits": 31,
                    "disk_writes": 24,
                },
                "cache_totals": {
                    "l2_tokens_on_disk": 5080,
                    "l2_block_tokens_on_disk": 1420,
                    "l2_ssm_tokens_on_disk": 3660,
                    "l2_tokens_on_disk_store_sum": 5080,
                },
                "ssm_companion": {
                    "disk": {
                        "enabled": True,
                        "entries": 5,
                        "total_tokens_on_disk": 3660,
                        "total_cached_tokens": 3660,
                        "stores": 5,
                    }
                },
            },
            "cacheHitTokens": 1919,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "streamTrace": [
            {
                "messageId": "lfm-stream-1",
                "count": 2,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed one",
            },
            {
                "messageId": "lfm-stream-2",
                "count": 2,
                "firstFullContent": "t",
                "lastFullContent": "ok streamed two",
            },
        ],
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "real_ui_tool_probe_1.txt REAL_UI_LIVE_TOOL_ONE",
                },
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "real_ui_tool_probe_2.txt REAL_UI_LIVE_TOOL_TWO",
                },
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=lfm "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "requestContract": {
            "requestMaxTokens": 512,
            "wireApi": "responses",
            "builtinToolsEnabled": True,
        },
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "tool_l2_cache_integrated" not in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" in lfm25["missing_surfaces"]


def _lfm_integrated_matrix_proof() -> dict[str, object]:
    return {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": [
            "start electron app",
            "[CHAT] Response complete: 134 tokens in 1.1s "
            "(356.4 t/s, live=228.3 t/s, TTFT: 0.07s, pp: 725 tokens "
            "(618 cached), 10984.8 pp/s, usage=server)",
            "[CHAT] Response complete: 136 tokens in 1.1s "
            "(353.2 t/s, live=228.3 t/s, TTFT: 0.07s, pp: 1024 tokens "
            "(917 cached), 14027.4 pp/s, usage=server)",
        ],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": [
                        "attention_kv",
                        "ssm_companion_state",
                        "async_rederive",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "hybrid_ssm_state",
                    },
                    "attention_kv_storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "attention_kv_layers_only",
                        "ssm_policy": "native_companion_state",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+ssm",
                        "cached_tokens": 917,
                    }
                },
            }
        },
        "chat": {
            "turns": [
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
            ],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 3},
                "block_disk_cache": {
                    "blocks_on_disk": 24,
                    "disk_hits": 31,
                    "disk_writes": 24,
                },
                "cache_totals": {
                    "l2_tokens_on_disk": 5080,
                    "l2_block_tokens_on_disk": 1420,
                    "l2_ssm_tokens_on_disk": 3660,
                },
                "ssm_companion": {
                    "disk": {
                        "enabled": True,
                        "entries": 5,
                        "total_tokens_on_disk": 3660,
                        "total_cached_tokens": 3660,
                        "stores": 5,
                    }
                },
            },
            "cacheHitTokens": 1919,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "streamTrace": [
            {
                "messageId": "lfm-stream-1",
                "count": 2,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed one",
            },
            {
                "messageId": "lfm-stream-2",
                "count": 2,
                "firstFullContent": "t",
                "lastFullContent": "ok streamed two",
            },
        ],
        "persistedToolCount": 24,
        "persistedToolsByMessage": [
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "real_ui_tool_probe_1.txt REAL_UI_LIVE_TOOL_ONE",
                },
                *({"phase": "result", "toolName": "run_command"} for _ in range(7)),
            ],
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": "real_ui_tool_probe_2.txt REAL_UI_LIVE_TOOL_TWO",
                },
                *({"phase": "result", "toolName": "run_command"} for _ in range(7)),
            ],
        ],
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
        },
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "serverCacheControls": {"verified": True},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=lfm "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "requestContract": {
            "requestMaxTokens": 512,
            "wireApi": "responses",
            "builtinToolsEnabled": True,
        },
    }


def test_release_regression_manifest_real_ui_integrated_tool_l2_requires_same_artifact_speed_floor():
    def lfm_base_proof() -> dict[str, object]:
        return {
            "modelName": "LFM2.5-8B-A1B-MXFP4",
            "appLogTail": ["start electron app"],
            "server": {
                "health": {
                    "status": "healthy",
                    "model_loaded": True,
                    "native_cache": {
                        "family": "lfm2_moe",
                        "schema": "hybrid_ssm_v1",
                        "cache_type": "hybrid_ssm_typed",
                        "components": [
                            "attention_kv",
                            "ssm_companion_state",
                            "async_rederive",
                        ],
                        "generic_turboquant_kv": {
                            "enabled": False,
                            "reason": "hybrid_ssm_state",
                        },
                        "attention_kv_storage_quantization": {
                            "enabled": True,
                            "mode": "storage_boundary",
                            "bits": 4,
                            "group_size": 64,
                            "applies_to": "attention_kv_layers_only",
                            "ssm_policy": "native_companion_state",
                        },
                        "prefix": True,
                        "paged": True,
                        "block_disk_l2": True,
                    },
                    "scheduler": {
                        "last_cache_execution": {
                            "cache_detail": "paged+ssm",
                            "cached_tokens": 917,
                        }
                    },
                }
            },
            "chat": {
                "turns": [
                    {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_ONE"},
                    {"role": "assistant", "content": "REAL_UI_LIVE_TOOL_TWO"},
                ],
                "rawParserTagLeak": False,
                "cjkLeakCount": 0,
                "koreanLeakCount": 0,
            },
            "cache": {
                "before": {"scheduler_cache": {"hits": 0}},
                "after": {
                    "scheduler_cache": {"hits": 3},
                    "block_disk_cache": {
                        "blocks_on_disk": 24,
                        "disk_hits": 31,
                        "disk_writes": 24,
                    },
                    "cache_totals": {
                        "l2_tokens_on_disk": 5080,
                        "l2_block_tokens_on_disk": 1420,
                        "l2_ssm_tokens_on_disk": 3660,
                    },
                },
                "cacheHitTokens": 1919,
            },
            "rendererWireApi": "responses",
            "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
            "streamTrace": [
                {
                    "messageId": "lfm-stream-1",
                    "count": 2,
                    "firstFullContent": "o",
                    "lastFullContent": "ok streamed one",
                },
                {
                    "messageId": "lfm-stream-2",
                    "count": 2,
                    "firstFullContent": "t",
                    "lastFullContent": "ok streamed two",
                },
            ],
            "persistedToolCount": 24,
            "persistedToolsByMessage": [
                [
                    {"phase": "calling", "toolName": "run_command"},
                    {"phase": "executing", "toolName": "run_command"},
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": "real_ui_tool_probe_1.txt REAL_UI_LIVE_TOOL_ONE",
                    },
                ],
                [
                    {"phase": "calling", "toolName": "run_command"},
                    {"phase": "executing", "toolName": "run_command"},
                    {
                        "phase": "result",
                        "toolName": "run_command",
                        "detail": "real_ui_tool_probe_2.txt REAL_UI_LIVE_TOOL_TWO",
                    },
                ],
            ],
            "toolProbeFiles": {
                "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
                "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
            },
            "requestedBuiltinTools": True,
            "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
            "serverCacheControls": {"verified": True},
            "serverLogTail": [
                "INFO:vmlx_engine.server:Resolved sampling kwargs "
                "route=/v1/responses model=lfm "
                "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
            ],
            "requestContract": {
                "requestMaxTokens": 512,
                "wireApi": "responses",
                "builtinToolsEnabled": True,
            },
        }

    integrated_without_speed = lfm_base_proof()
    speed_only = lfm_base_proof()
    speed_only["eventCounts"] = {"complete": 2, "stream": 4, "tool": 0}
    speed_only["persistedToolCount"] = 0
    speed_only["persistedToolsByMessage"] = []
    speed_only["toolProbeFiles"] = {}
    speed_only["appLogTail"] = [
        "start electron app",
        "[CHAT] Response complete: 134 tokens in 1.1s "
        "(356.4 t/s, live=228.3 t/s, TTFT: 0.07s, pp: 725 tokens "
        "(618 cached), 10984.8 pp/s, usage=server)",
        "[CHAT] Response complete: 136 tokens in 1.1s "
        "(353.2 t/s, live=228.3 t/s, TTFT: 0.07s, pp: 1024 tokens "
        "(917 cached), 14027.4 pp/s, usage=server)",
    ]

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "lfm25_moe_a1b_responses_delta": integrated_without_speed,
                "lfm25_moe_a1b_responses": speed_only,
            },
        }
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "live_speed_floor" in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" not in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_integrated_tool_l2_requires_same_artifact_policy_and_defaults():
    integrated_missing_policy = _lfm_integrated_matrix_proof()
    integrated_missing_policy.pop("serverLogTail")
    native_cache = integrated_missing_policy["server"]["health"]["native_cache"]
    native_cache["family"] = "plain_kv"
    native_cache["schema"] = "plain_kv_v1"
    native_cache["cache_type"] = "paged_kv"
    native_cache["components"] = ["attention_kv"]
    native_cache.pop("attention_kv_storage_quantization")

    policy_only = _lfm_integrated_matrix_proof()
    policy_only["eventCounts"] = {"complete": 2, "stream": 4, "tool": 0}
    policy_only["persistedToolCount"] = 0
    policy_only["persistedToolsByMessage"] = []
    policy_only["toolProbeFiles"] = {}

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "lfm25_moe_a1b_responses_delta": integrated_missing_policy,
                "lfm25_moe_a1b_responses": policy_only,
            },
        }
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "generation_defaults_applied" in lfm25["covered_surfaces"]
    assert "architecture_cache_policy" in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" not in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_integrated_tool_l2_requires_same_artifact_delta_streaming():
    integrated_missing_delta = _lfm_integrated_matrix_proof()
    integrated_missing_delta.pop("streamTrace")

    delta_only = _lfm_integrated_matrix_proof()
    delta_only["cache"] = {
        "before": {"scheduler_cache": {"hits": 0}},
        "after": {"scheduler_cache": {"hits": 0}},
        "cacheHitTokens": 0,
    }
    delta_only["eventCounts"] = {"complete": 2, "stream": 4, "tool": 0}
    delta_only["persistedToolCount"] = 0
    delta_only["persistedToolsByMessage"] = []
    delta_only["toolProbeFiles"] = {}
    delta_only["server"]["health"].pop("native_cache")
    delta_only["server"]["health"].pop("kv_cache_quantization")

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "lfm25_moe_a1b_responses": integrated_missing_delta,
                "lfm25_moe_a1b_responses_delta": delta_only,
            },
        }
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "responses_delta_streaming" in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" not in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_integrated_tool_l2_requires_same_artifact_extensive_tool_churn():
    minimal_tool_churn = _lfm_integrated_matrix_proof()
    minimal_tool_churn["eventCounts"]["tool"] = 3
    minimal_tool_churn["persistedToolCount"] = 3

    matrix = _validate_current_real_ui_live_model_matrix(
        {
            "status": "pass",
            "proofs": {
                "lfm25_moe_a1b_responses_delta": minimal_tool_churn,
            },
        }
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "long_tool_loop" in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" not in lfm25["covered_surfaces"]
    assert "tool_l2_cache_integrated" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_requires_lfm_live_speed_floor():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": [
                        "attention_kv",
                        "ssm_companion_state",
                        "async_rederive",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "hybrid_ssm_state",
                    },
                    "attention_kv_storage_quantization": {
                        "enabled": True,
                        "mode": "storage_boundary",
                        "bits": 4,
                        "group_size": 64,
                        "applies_to": "attention_kv_layers_only",
                        "ssm_policy": "native_companion_state",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+ssm",
                        "cached_tokens": 917,
                    }
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {"cacheHitTokens": 917},
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "streamTrace": [
            {
                "messageId": "lfm-stream",
                "count": 4,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed",
            }
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "serverLogTail": [
            "INFO:vmlx_engine.server:Resolved sampling kwargs "
            "route=/v1/responses model=lfm "
            "kwargs={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512}"
        ],
        "requestContract": {"requestMaxTokens": 512, "wireApi": "responses"},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert (
        "live_speed_floor"
        in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY["lfm25"]
    )
    assert "live_speed_floor" not in lfm25["covered_surfaces"]
    assert "live_speed_floor" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_accepts_structured_live_speed_samples():
    proof = {
        "modelName": "Step-3.7-Flash-JANG_2L",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "mixed_attention",
                    "schema": "mixed_swa_kv_v1",
                    "cache_type": "mixed_swa_kv",
                    "components": [
                        "full_attention_kv",
                        "sliding_window_kv",
                        "rotating_window_metadata",
                    ],
                    "generic_turboquant_kv": {
                        "enabled": False,
                        "reason": "not_active",
                    },
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
                "kv_cache_quantization": {
                    "enabled": True,
                    "bits": 4,
                    "group_size": 64,
                },
                "scheduler": {
                    "last_cache_execution": {
                        "cache_detail": "paged+mixed_swa",
                        "cached_tokens": 2612,
                    }
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {"cacheHitTokens": 2612},
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "stream": 4, "tool": 24},
        "streamTrace": [
            {
                "messageId": "step-stream",
                "count": 4,
                "firstFullContent": "o",
                "lastFullContent": "ok streamed",
            }
        ],
        "liveSpeedSamples": [
            {"tokens": 78, "liveTokensPerSecond": 50.2, "ttftSeconds": 0.43},
            {"tokens": 86, "liveTokensPerSecond": 49.9, "ttftSeconds": 0.44},
        ],
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True, "maxTokens": 512},
        "requestContract": {"requestMaxTokens": 512, "wireApi": "responses"},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"step37_flash_jang2l": proof}}
    )

    step37 = matrix["covered_families"]["step37"]
    assert "live_speed_floor" in step37["covered_surfaces"]
    assert "live_speed_floor" not in step37["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_empty_tool_status_spam():
    proof = {
        "modelName": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "plain_kv",
                    "schema": "plain_kv_v1",
                    "cache_type": "paged_kv",
                    "components": ["attention_kv"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "ok"}],
            "rawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 49, "reasoningDone": 1},
        "persistedToolCount": 49,
        "persistedToolsByMessage": [
            [{"phase": "generating", "toolName": "", "iteration": 0} for _ in range(49)]
        ],
        "persistedReasoningCount": 1,
        "chatOverrides": {"builtinToolsEnabled": True},
        "requestedBuiltinTools": True,
        "serverCacheControls": {"verified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"gemma4": proof}}
    )

    gemma4 = matrix["covered_families"]["gemma4"]
    assert "long_tool_loop" not in gemma4["covered_surfaces"]
    assert "long_tool_loop" in gemma4["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_result_only_tool_lifecycle():
    proof = _lfm_integrated_matrix_proof()
    proof["persistedToolsByMessage"] = [
        [
            {
                "phase": "result",
                "toolName": "run_command",
                "detail": "real_ui_tool_probe_1.txt REAL_UI_LIVE_TOOL_ONE",
            },
        ],
        [
            {
                "phase": "result",
                "toolName": "run_command",
                "detail": "real_ui_tool_probe_2.txt REAL_UI_LIVE_TOOL_TWO",
            },
        ],
    ]

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25_moe_a1b_responses_delta": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "long_tool_loop" not in lfm25["covered_surfaces"]
    assert "long_tool_loop" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_failed_tool_loop():
    proof = {
        "modelName": "LFM2.5-8B-A1B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "lfm2_moe",
                    "schema": "hybrid_ssm_v1",
                    "cache_type": "hybrid_ssm_typed",
                    "components": ["attention_kv", "ssm_companion_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [{"role": "assistant", "content": "tool failed"}],
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 8},
        "persistedToolsByMessage": [
            [
                {"phase": "calling", "toolName": "run_command"},
                {"phase": "executing", "toolName": "run_command"},
                {
                    "phase": "error",
                    "toolName": "run_command",
                    "detail": "Exit code: 127\n/bin/sh: bad-command: command not found",
                },
                {"phase": "result", "toolName": "run_command"},
            ],
            [
                {"phase": "result", "toolName": "run_command"},
            ],
        ],
        "persistedToolCount": 5,
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"lfm25": proof}}
    )

    lfm25 = matrix["covered_families"]["lfm25"]
    assert "long_tool_loop" not in lfm25["covered_surfaces"]
    assert "long_tool_loop" in lfm25["missing_surfaces"]


def test_release_regression_manifest_real_ui_matrix_rejects_wrong_tool_probe_payload():
    proof = {
        "modelName": "ZAYA1-8B-MXFP4",
        "appLogTail": ["start electron app"],
        "server": {
            "health": {
                "status": "healthy",
                "model_loaded": True,
                "native_cache": {
                    "family": "zaya",
                    "schema": "zaya_cca_v1",
                    "cache_type": "hybrid_zaya_cca",
                    "components": ["attention_kv", "cca_conv_state"],
                    "prefix": True,
                    "paged": True,
                    "block_disk_l2": True,
                },
            }
        },
        "chat": {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "Use run_command to create real_ui_tool_probe_1.txt "
                        "with REAL_UI_LIVE_TOOL_ONE"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use run_command to read real_ui_tool_probe_1.txt, create "
                        "real_ui_tool_probe_2.txt, and write REAL_UI_LIVE_TOOL_TWO"
                    ),
                },
            ],
            "rawParserTagLeak": False,
            "reasoningRawParserTagLeak": False,
            "cjkLeakCount": 0,
            "koreanLeakCount": 0,
            "reasoningCjkLeakCount": 0,
            "reasoningKoreanLeakCount": 0,
            "reasoningNumericRunCount": 0,
        },
        "cache": {
            "before": {"scheduler_cache": {"hits": 0}},
            "after": {
                "scheduler_cache": {"hits": 1},
                "block_disk_cache": {"disk_hits": 1},
                "cache_totals": {"l2_tokens_on_disk": 12},
            },
            "cacheHitTokens": 12,
        },
        "rendererWireApi": "responses",
        "eventCounts": {"complete": 2, "tool": 8},
        "persistedToolsByMessage": [
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ printf %s REAL_UI_LIVE_TOOL_ONE > "
                        "real_ui_tool_probe_1.txt\n\n"
                    ),
                },
            ],
            [
                {
                    "phase": "result",
                    "toolName": "run_command",
                    "detail": (
                        "$ cat real_ui_tool_probe_1.txt > "
                        "real_ui_tool_probe_2.txt\n\n"
                    ),
                },
            ],
        ],
        "persistedToolCount": 2,
        "toolProbeFiles": {
            "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE\n",
            "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_ONE\n",
        },
        "requestedBuiltinTools": True,
        "chatOverrides": {"builtinToolsEnabled": True},
        "serverCacheControls": {"verified": True},
    }

    matrix = _validate_current_real_ui_live_model_matrix(
        {"status": "pass", "proofs": {"zaya_text": proof}}
    )

    zaya = matrix["covered_families"]["zaya_text"]
    assert "long_tool_loop" not in zaya["covered_surfaces"]
    assert "long_tool_loop" in zaya["missing_surfaces"]


def test_release_regression_manifest_validates_current_proof_sweep_artifacts(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    api_cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "noheavy-api-cache-endpoint-runtime"
    ]
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    panel_tool_security_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "panel-tool-security-loop-boundary"
    ]
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    jang_model_compat_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "jang-model-compat-runtime-boundary"
    ]
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    open_requirement_details = _passing_open_requirement_details()
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == model_family_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == max_output_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS
                        },
                        "missing_markers": [],
                        "failed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == packaged_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS
                        },
                        "failed": [],
                        "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == release_surface_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == model_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True
                            for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == parser_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == generation_artifact:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True
                            for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == reasoning_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == tool_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == panel_tool_security_artifact:
            path.write_text(
                json.dumps({"status": "pass", "returncode": 0}) + "\n",
                encoding="utf-8",
            )
        elif artifact == mtp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == vl_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS
                        },
                        "missing_engine_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mcp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_requirement_details,
                "dsv4_proof_artifact_freshness_evidence": list(
                    CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS
                ),
                **_passing_current_suite_progress_checkpoints(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    _write_expected_diagnostic_live_smoke_artifacts(tmp_path)
    _write_current_objective_digest(tmp_path)
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_dsv4_source_memory_preflight_artifact(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)
    _write_passing_issue179_minimax_k_live_probe_memory_preflight(tmp_path)
    _write_passing_step37_vlm_runtime_audit(tmp_path)
    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)

    result = validate_current_proof_sweep_artifacts(tmp_path)

    bad_components = {
        key: value
        for key, value in result.items()
        if isinstance(value, dict)
        and (
            value.get("failures")
            or value.get("missing")
            or value.get("not_pass")
            or value.get("status") not in {"pass", "open"}
        )
    }
    assert result["status"] == "fail", json.dumps(bad_components, indent=2, sort_keys=True)
    assert result["failed_components"] == ["no_open_objective_requirements"]
    assert result["component_ok"]["no_release_blockers"] is True
    assert result["component_ok"]["packaged_app_developer_id_signing"] is True
    assert result["component_ok"]["dsv4_long_output_code_exactness"] is True
    assert result["component_ok"]["no_open_objective_requirements"] is False
    assert result["component_ok"]["real_ui_full_model_matrix"] is True
    assert result["component_ok"]["real_ui_dsv4"] is True
    assert result["component_ok"]["objective_digest"] is True
    assert result["component_ok"]["mimo_v2_jang2l_root_cause"] is True
    assert result["mimo_v2_jang2l_root_cause"]["status"] == "pass"
    assert result["objective_digest"]["status"] == "pass"
    assert result["missing"] == []
    assert result["not_pass"] == []

    assert result["regression_suite"] == {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "pass",
        "failed_steps": [],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "open_requirement_details": open_requirement_details,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "open_requirement_detail_failures": [],
    }
    assert result["model_family_matrix"] == {
        "artifact": model_family_artifact,
        "status": "pass",
        "matched_rows": list(EXPECTED_CURRENT_MODEL_FAMILY_ROWS),
        "missing_rows": [],
        "missing_expected_rows": [],
    }
    assert result["model_artifact_matrix"] == {
        "artifact": model_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["cache_architecture_matrix"] == {
        "artifact": cache_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS},
        "missing_markers": [],
        "missing_api_checks": [],
        "missing_api_command_markers": [],
        "missing_panel_markers": [],
        "missing_family_rows": [],
        "failed_family_rows": [],
        "cache_family_matrix": _passing_cache_family_matrix(),
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["parser_registry_matrix"] == {
        "artifact": parser_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["generation_defaults_matrix"] == {
        "artifact": generation_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS},
        "missing_markers": [],
        "missing_family_rows": [],
        "failed_family_rows": [],
        "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["api_surface_matrix"] == {
        "artifact": api_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS},
        "missing_nested_checks": [],
        "missing_nested_markers": [],
        "missing_panel_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["reasoning_template_matrix"] == {
        "artifact": reasoning_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["tool_call_matrix"] == {
        "artifact": tool_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["native_mtp_matrix"] == {
        "artifact": mtp_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["vl_media_matrix"] == {
        "artifact": vl_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS},
        "missing_engine_markers": [],
        "missing_panel_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["mcp_policy_matrix"] == {
        "artifact": mcp_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["max_output_context_matrix"] == {
        "artifact": max_output_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS},
        "missing_markers": [],
        "failed": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["packaged_integrity_matrix"] == {
        "artifact": packaged_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS},
        "failed": [],
        "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "package_signing_preflight": {},
        "release_blockers": [],
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["release_surface_matrix"] == {
        "artifact": release_surface_artifact,
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS},
        "failed_checks": [],
        "missing_expected_checks": [],
        "informational_false_checks": [],
    }
    assert result["live_smoke_summaries"]["status"] == "pass"
    assert result["live_smoke_summaries"]["non_mimo_status"] == "pass"
    assert (
        "zaya_text_mxfp4"
        in result["live_smoke_summaries"]["artifacts"]
    )
    assert (
        result["live_smoke_summaries"]["artifacts"]["zaya_text_mxfp4"]["artifact"]
        == "build/current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json"
    )
    assert result["live_smoke_summaries"]["missing"] == []
    assert [
        item["artifact"] for item in result["live_smoke_summaries"]["not_pass"]
    ] == []
    assert result["live_tool_smoke_summaries"]["status"] == "pass"
    assert result["live_tool_smoke_summaries"]["non_mimo_status"] == "pass"
    assert result["live_tool_smoke_summaries"]["missing"] == []
    assert [
        item["artifact"]
        for item in result["live_tool_smoke_summaries"]["not_pass"]
    ] == []
    assert (
        result["diagnostic_live_smoke_summaries"]
        == _expected_passing_diagnostic_live_smoke_summaries()
    )
    assert result["dev_ui_proof"] == {
        "status": "pass",
        "artifacts": dict(CURRENT_DEV_UI_PROOF_ARTIFACTS),
        "missing": [],
        "failures": [],
    }
    assert result["real_ui_live_model_proof"]["status"] == "pass"
    assert result["real_ui_live_model_proof"]["artifacts"] == dict(
        CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS
    )
    assert result["real_ui_live_model_proof"]["missing"] == []
    assert result["real_ui_live_model_proof"]["failures"] == []
    assert set(result["real_ui_live_model_proof"]["proofs"]) == set(
        CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS
    )
    assert result["real_ui_live_model_matrix"]["unblocked_non_mimo_status"] == "pass"
    assert result["real_ui_live_model_matrix"][
        "unblocked_non_mimo_missing_families"
    ] == []
    assert result["real_ui_live_model_matrix"][
        "unblocked_non_mimo_partial_families"
    ] == []
    assert result["component_ok"]["real_ui_unblocked_non_mimo"] is True
    assert result["component_ok"]["jang_model_compat_matrix"] is True
    assert result["component_ok"]["panel_tool_security_matrix"] is True
    assert result["component_ok"]["noheavy_api_cache_matrix"] is True
    assert result["noheavy_api_cache_matrix"] == {
        "artifact": CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
            "noheavy-api-cache-endpoint-runtime"
        ],
        "status": "pass",
        "checks": {name: True for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS},
        "missing_markers": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }
    assert result["jang_model_compat_matrix"] == {
        "artifact": CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
            "jang-model-compat-runtime-boundary"
        ],
        "status": "pass",
        "failed": [],
        "missing": [],
    }
    assert result["panel_tool_security_matrix"] == {
        "artifact": CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
            "panel-tool-security-loop-boundary"
        ],
        "status": "pass",
        "returncode": 0,
        "missing": [],
    }
    assert result["installed_app_runtime_parity_audit"] == {
        "artifact": CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }
    assert result["issue175_177_installed_runtime_audit"] == {
        "artifact": CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }
    assert result["issue175_177_live_runtime_audit"] == {
        "artifact": CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT,
        "status": "pass",
        "missing": [],
        "failures": [],
    }
    assert result["mimo_v2_jang2l_root_cause"]["status"] == "pass"
    assert result["component_ok"]["mimo_v2_jang2l_root_cause"] is True
    ledger = result["release_blocker_ledger"]
    assert ledger["status"] == "pass"
    assert ledger["deferred_release_families"] == [
        {
            "family": "dsv4",
            "reason": "deferred_per_20260602_emergency_release_scope",
        }
    ]
    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert blockers == {}
    deferred = {gap["id"]: gap for gap in ledger["deferred_release_gaps"]}
    assert set(deferred) >= {
        "dsv4_long_output_code_exactness_open",
        "real_ui_dsv4_memory_blocked",
    }

    dsv4_blocker = deferred["dsv4_long_output_code_exactness_open"]
    assert dsv4_blocker["status"] == "deferred"
    assert dsv4_blocker["evidence"] == CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT
    assert (
        dsv4_blocker["next_proof"]
        == "Run and pass DSV4 long-output/code exactness with current source/app in a memory-safe local session."
    )
    dsv4_details = dsv4_blocker["details"]
    assert dsv4_details["artifact_present"] is True
    assert dsv4_details["status"] == "skipped"
    assert dsv4_details["reason"] == "insufficient_vm_stat_memory"
    assert (
        dsv4_details["model"]
        == "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"
    )
    assert dsv4_details["preflight_memory_source"] == (
        "vm_stat_free_plus_speculative_purgeable"
    )
    assert dsv4_details["did_not_launch"] is True
    assert dsv4_details["launch_decision"] == "do_not_launch"
    assert dsv4_details["launch_allowed"] is False
    assert dsv4_details["launch_blockers"] == ["insufficient_memory"]
    assert dsv4_details["active_heavy_process_count"] == 0
    assert dsv4_details["active_heavy_processes"] == []
    assert dsv4_details["case_count"] == 14
    assert dsv4_details["selected_cases"] == [
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
    assert dsv4_details["available_for_gate_gb"] < dsv4_details["required_available_gb"]
    assert dsv4_details["memory_gap_gb"] > 0
    assert dsv4_details["strict_vm_stat_memory_gap_gb"] > 0
    assert dsv4_details["top_memory_processes"]
    assert "memory" in dsv4_details["commands"]
    assert "memory_pressure" in dsv4_details["commands"]

    real_ui_blocker = deferred["real_ui_dsv4_memory_blocked"]
    assert real_ui_blocker["status"] == "deferred"
    assert real_ui_blocker["evidence"] == CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    assert (
        real_ui_blocker["next_proof"]
        == "Run DSV4 real Electron UI proof on a local machine/session with enough free memory."
    )
    real_ui_details = real_ui_blocker["details"]
    assert real_ui_details["artifact"] == CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    assert real_ui_details["reason"] == "insufficient_memory"
    assert (
        real_ui_details["model_path"]
        == "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"
    )
    assert real_ui_details["preflight_memory_source"] == (
        "vm_stat_free_plus_speculative_purgeable"
    )
    assert real_ui_details["did_not_launch"] is True
    assert real_ui_details["launch_decision"] == "do_not_launch"
    assert real_ui_details["launch_allowed"] is False
    assert real_ui_details["launch_blockers"] == ["insufficient_memory"]
    assert real_ui_details["active_heavy_process_count"] == 0
    assert real_ui_details["active_heavy_processes"] == []
    assert real_ui_details["available_for_gate_gb"] < real_ui_details["required_available_gb"]
    assert real_ui_details["memory_gap_gb"] > 0
    assert real_ui_details["top_memory_processes"]


def test_release_blocker_ledger_tracks_mimo_as_current_release_blocker():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "open"},
        mimo_v2_jang2l_root_cause={
            "status": "open",
            "local_release_clearance": False,
            "artifacts": {"length_sweep": CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT},
            "prompt_length_coherence_blocked": True,
        },
        issue175_179_release_boundary_audit={"status": "open", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "open"},
        real_ui_live_model_matrix={"status": "open", "missing_families": []},
    )

    blocker_ids = [blocker["id"] for blocker in ledger["blockers"]]
    assert "mimo_v2_jang2l_runtime_quality_open" in blocker_ids
    assert not any(
        item["family"] == "mimo_v2"
        for item in ledger["deferred_release_families"]
    )
    assert ledger["deferred_release_families"] == [{"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"}]


def test_release_blocker_ledger_tracks_public_version_collision():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        release_surface_matrix={
            "status": "pass",
            "failed_checks": ["staged_source_version_not_public"],
            "artifact": "build/current-release-surface-contract.json",
        },
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"status": "pass"},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert blockers["source_version_already_public"]["evidence"] == (
        "build/current-release-surface-contract.json"
    )


def test_release_blocker_ledger_does_not_use_remote_max2_artifacts_as_release_evidence():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "open"},
        mimo_v2_jang2l_root_cause={
            "status": "open",
            "local_release_clearance": False,
            "remote_evidence_only": True,
            "remote_artifacts": [
                "remote-max2:mimo-v25-tp4-live-proof.md",
            ],
            "artifacts": {"length_sweep": CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT},
        },
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    assert ledger["status"] == "open"
    for blocker in ledger["blockers"]:
        assert "remote-max2" not in blocker["evidence"]
    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert set(blockers) == {"mimo_v2_jang2l_runtime_quality_open"}
    assert ledger["deferred_release_families"] == [{"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"}]


def test_release_blocker_ledger_tracks_missing_local_mimo_root_cause_artifacts():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={
            "status": "missing",
            "remote_evidence_only": False,
            "missing": [
                CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
                CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT,
            ],
        },
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    assert "mimo_v2_jang2l_runtime_quality_open" in blockers
    assert ledger["deferred_release_families"] == [{"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"}]


def test_release_blocker_ledger_reports_mimo_exactness_next_proof_without_stale_rows():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={
            "status": "open",
            "local_release_clearance": False,
            "artifact_exactness_blocked": True,
            "prompt_length_coherence_blocked": False,
            "tool_protocol_blocked": False,
            "decode_speed_target_blocked": False,
            "media_unwired": False,
            "source_vs_quant_requirement_satisfied": True,
            "artifact_exactness_bundle_status": {
                "jangtq2": {
                    "blocked": True,
                    "classification": "jangtq2_plain_literal_copy_fails_before_parser_or_json_repair",
                    "evidence": CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
                },
                "jang2l": {
                    "blocked": True,
                    "classification": "jang2l_json_sentinel_semantic_mismatch_open",
                    "evidence": CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
                },
            },
            "release_boundary": (
                "local artifact/runtime has narrow text-cache proof but still "
                "fails current JANGTQ2 artifact exactness, current JANG_2L "
                "artifact exactness; do not release-clear MiMo"
            ),
        },
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blockers = {blocker["id"]: blocker for blocker in ledger["blockers"]}
    next_proof = blockers["mimo_v2_jang2l_runtime_quality_open"]["next_proof"]
    assert "JANGTQ2 artifact exactness" in next_proof
    assert "JANG_2L artifact exactness" in next_proof
    assert "long-prompt" not in next_proof
    assert "tool protocol" not in next_proof
    assert "cache" not in next_proof


def test_release_blocker_ledger_tracks_packaged_signing_blocker():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
        packaged_integrity_matrix={
            "package_signing_preflight": {
                "signing_blocker_reason": "developer_id_keychain_user_interaction_not_allowed",
                "developer_id_signed": False,
                "developer_id_identity_count": 9,
                "signature_is_adhoc": True,
                "hardened_runtime_enabled": False,
                "team_identifier": "not set",
                "codesign_display_rc": 0,
                "signature_summary_tail": [
                    "Signature=adhoc",
                    "TeamIdentifier=not set",
                ],
                "simple_developer_id_sign_rc": 1,
                "simple_developer_id_sign_tail": [
                    "codesign-probe: User interaction is not allowed.",
                ],
                "codesign_verify_rc": 1,
                "verify_tail": ["adhoc verify failed"],
                "keychain_info_statuses": [
                    {
                        "keychain": "/Users/eric/Library/Keychains/vmlx-build.keychain-db",
                        "returncode": 36,
                        "tail": [
                            "security: SecKeychainCopySettings: User interaction is not allowed.",
                        ],
                    },
                ],
                "manual_remediation_required": True,
                "remediation_summary": "Developer ID identities are visible, but codesign cannot use the private key from this non-interactive process.",
                "remediation_steps": [
                    "Unlock the signing keychain in an interactive macOS session.",
                    "Grant codesign access to the Developer ID private key, for example with security set-key-partition-list using the keychain password outside Codex logs.",
                    "Rerun the packaged integrity contract and require package_signing_preflight.status=pass before notarization.",
                ],
            },
            "release_blockers": [
                {
                    "id": "packaged_app_developer_id_signing_blocked",
                    "status": "open",
                    "evidence": "package_signing_preflight",
                    "next_proof": "Build and verify a Developer ID signed vMLX.app before notarization.",
                }
            ]
        },
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "packaged_app_developer_id_signing_blocked",
                "status": "open",
                "evidence": "package_signing_preflight",
                "details": {
                    "signing_blocker_reason": "developer_id_keychain_user_interaction_not_allowed",
                    "developer_id_signed": False,
                    "developer_id_identity_count": 9,
                    "signature_is_adhoc": True,
                    "hardened_runtime_enabled": False,
                    "team_identifier": "not set",
                    "codesign_display_rc": 0,
                    "signature_summary_tail": [
                        "Signature=adhoc",
                        "TeamIdentifier=not set",
                    ],
                    "simple_developer_id_sign_rc": 1,
                    "simple_developer_id_sign_tail": [
                        "codesign-probe: User interaction is not allowed.",
                    ],
                    "codesign_verify_rc": 1,
                    "verify_tail": ["adhoc verify failed"],
                    "keychain_info_statuses": [
                        {
                            "keychain": "/Users/eric/Library/Keychains/vmlx-build.keychain-db",
                            "returncode": 36,
                            "tail": [
                                "security: SecKeychainCopySettings: User interaction is not allowed.",
                            ],
                        },
                    ],
                    "manual_remediation_required": True,
                    "remediation_summary": "Developer ID identities are visible, but codesign cannot use the private key from this non-interactive process.",
                    "remediation_steps": [
                        "Unlock the signing keychain in an interactive macOS session.",
                        "Grant codesign access to the Developer ID private key, for example with security set-key-partition-list using the keychain password outside Codex logs.",
                        "Rerun the packaged integrity contract and require package_signing_preflight.status=pass before notarization.",
                    ],
                },
                "next_proof": "Build and verify a Developer ID signed vMLX.app before notarization.",
            }
        ],
        "deferred_release_gaps": [],
        "deferred_release_families": [{"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"}],
    }


def test_release_blocker_ledger_defers_dsv4_exactness_open_requirement():
    ledger = _current_release_blocker_ledger(
        regression_suite={
            "open_requirements": [
                "DSV4 long-output/code/file-generation quality is release-cleared"
            ]
        },
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    assert ledger["status"] == "pass"
    assert ledger["blockers"] == []
    assert ledger["deferred_release_families"] == [
        {"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"},
    ]
    deferred = {gap["id"]: gap for gap in ledger["deferred_release_gaps"]}
    assert deferred["dsv4_long_output_code_exactness_open"] == {
        "id": "dsv4_long_output_code_exactness_open",
        "status": "deferred",
        "evidence": CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
        "reason": "deferred_per_20260602_emergency_release_scope",
        "next_proof": "Run and pass DSV4 long-output/code exactness with current source/app in a memory-safe local session.",
    }


def test_release_blocker_ledger_preserves_deferred_dsv4_exactness_memory_pressure_diagnostic():
    ledger = _current_release_blocker_ledger(
        regression_suite={
            "open_requirements": [
                "DSV4 long-output/code/file-generation quality is release-cleared"
            ],
            "open_requirement_details": {
                "DSV4 long-output/code/file-generation quality is release-cleared": {
                    "details": {
                        "current_source_full_output_preflight": {
                            "artifact_present": True,
                            "status": "skipped",
                            "reason": "insufficient_vm_stat_memory",
                            "model": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
                            "commands": {
                                "memory": "vm_stat",
                                "memory_pressure": "memory_pressure",
                            },
                            "available_gb": 107.63,
                            "required_available_gb": 120.0,
                            "memory_gap_gb": 61.74,
                            "strict_vm_stat_memory_gap_gb": 61.74,
                            "psutil_available_gap_gb": 12.37,
                            "free_plus_speculative_purgeable_gb": 58.26,
                            "memory_pressure_free_percent": 94,
                            "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                            "did_not_launch": True,
                            "launch_decision": "do_not_launch",
                            "launch_blockers": ["insufficient_memory"],
                            "active_heavy_process_count": 0,
                            "active_heavy_processes": [],
                            "selected_cases": ["chat_max", "responses_on"],
                            "case_count": 14,
                        }
                    }
                }
            },
        },
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blocker = {
        gap["id"]: gap for gap in ledger["deferred_release_gaps"]
    }["dsv4_long_output_code_exactness_open"]
    assert blocker["status"] == "deferred"
    assert blocker["details"]["artifact_present"] is True
    assert blocker["details"]["status"] == "skipped"
    assert blocker["details"]["reason"] == "insufficient_vm_stat_memory"
    assert (
        blocker["details"]["model"]
        == "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"
    )
    assert blocker["details"]["commands"] == {
        "memory": "vm_stat",
        "memory_pressure": "memory_pressure",
    }
    assert blocker["details"]["free_plus_speculative_purgeable_gb"] == 58.26
    assert blocker["details"]["memory_pressure_free_percent"] == 94
    assert blocker["details"]["active_heavy_processes"] == []
    assert blocker["details"]["selected_cases"] == ["chat_max", "responses_on"]


def test_release_blocker_ledger_prefers_current_dsv4_exactness_preflight_artifact(
    tmp_path,
):
    artifact = tmp_path / CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason": "insufficient_vm_stat_memory",
                "model": "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520",
                "commands": {
                    "memory": "vm_stat",
                    "memory_pressure": "memory_pressure",
                },
                "available_gb": 105.0,
                "required_available_gb": 120.0,
                "required_free_gb": 120.0,
                "min_free_gb": 120.0,
                "available_for_gate_gb": 79.25,
                "memory_gap_gb": 40.75,
                "strict_vm_stat_memory_gap_gb": 40.75,
                "psutil_available_gap_gb": 15.0,
                "free_plus_speculative_purgeable_gb": 79.25,
                "memory_pressure_free_percent": 95,
                "preflight_memory_source": "vm_stat_free_plus_speculative_purgeable",
                "did_not_launch": True,
                "launch_decision": "do_not_launch",
                "launch_allowed": False,
                "launch_blockers": ["insufficient_memory"],
                "active_heavy_process_count": 0,
                "active_heavy_processes": [],
                "top_memory_processes": [
                    {"pid": 4321, "rss_gb": 2.0, "command": "current-codex"}
                ],
                "selected_cases": ["chat_off", "responses_off"],
                "case_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = _current_release_blocker_ledger(
        root=tmp_path,
        regression_suite={
            "open_requirements": [
                "DSV4 long-output/code/file-generation quality is release-cleared"
            ],
            "open_requirement_details": {
                "DSV4 long-output/code/file-generation quality is release-cleared": {
                    "details": {
                        "current_source_full_output_preflight": {
                            "status": "skipped",
                            "reason": "stale_suite_detail",
                            "available_for_gate_gb": 12.0,
                            "memory_gap_gb": 108.0,
                            "top_memory_processes": [
                                {
                                    "pid": 9999,
                                    "rss_gb": 99.0,
                                    "command": "stale-runner",
                                }
                            ],
                        }
                    }
                }
            },
        },
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    blocker = {
        gap["id"]: gap for gap in ledger["deferred_release_gaps"]
    }["dsv4_long_output_code_exactness_open"]
    assert blocker["status"] == "deferred"
    assert blocker["details"]["reason"] == "insufficient_vm_stat_memory"
    assert blocker["details"]["available_for_gate_gb"] == 79.25
    assert blocker["details"]["memory_gap_gb"] == 40.75
    assert blocker["details"]["top_memory_processes"] == [
        {"pid": 4321, "rss_gb": 2.0, "command": "current-codex"}
    ]


def test_release_blocker_ledger_tracks_stale_real_ui_request_contract_proofs():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_proof={
            "status": "fail",
            "failures": [
                "request_contract_missing:gemma4",
                "request_contract_incomplete:qwen36:enableThinking",
            ],
        },
        real_ui_live_model_matrix={"status": "pass", "missing_families": []},
    )

    assert ledger["blockers"] == [
        {
            "id": "real_ui_request_contract_proofs_stale",
            "status": "open",
            "evidence": "current_proof_sweep.real_ui_live_model_proof.failures:request_contract_incomplete:qwen36:enableThinking,request_contract_missing:gemma4",
            "next_proof": "Rerun real Electron UI model proofs with current live-real-ui-model-proof.mjs so every artifact records its prompts and request knobs.",
        }
    ]


def test_release_blocker_ledger_splits_real_ui_missing_family_blockers():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )

    blocker_ids = [blocker["id"] for blocker in ledger["blockers"]]
    assert blocker_ids == ["real_ui_unblocked_non_mimo_missing"]
    assert "missing_families:mimo_v2" in ledger["blockers"][-1]["evidence"]
    deferred = {gap["id"]: gap for gap in ledger["deferred_release_gaps"]}
    assert deferred["real_ui_dsv4_memory_blocked"]["status"] == "deferred"
    assert deferred["real_ui_dsv4_memory_blocked"]["evidence"] == (
        CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    )

    unblocked_missing_ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4", "step37", "lfm25"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )
    assert [blocker["id"] for blocker in unblocked_missing_ledger["blockers"]] == [
        "real_ui_unblocked_non_mimo_missing",
    ]
    assert {
        gap["id"] for gap in unblocked_missing_ledger["deferred_release_gaps"]
    } >= {"real_ui_dsv4_memory_blocked"}
    assert (
        "missing_families:lfm25,mimo_v2,step37"
        in unblocked_missing_ledger["blockers"][-1]["evidence"]
    )

    unblocked_partial_ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4"],
            "partial_families": ["lfm25"],
            "covered_families": {
                "lfm25": {
                    "status": "partial",
                    "missing_surfaces": ["l2_disk_storage"],
                    "artifact": "docs/internal/agent-notes/current-lfm25-proof.json",
                }
            },
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )
    assert [blocker["id"] for blocker in unblocked_partial_ledger["blockers"]] == [
        "real_ui_unblocked_non_mimo_missing",
        "real_ui_unblocked_non_mimo_partial",
    ]
    assert (
        "partial_families:lfm25"
        in unblocked_partial_ledger["blockers"][-1]["evidence"]
    )
    assert unblocked_partial_ledger["blockers"][-1]["details"] == {
        "partial_families": {
            "lfm25": {
                "missing_surfaces": ["l2_disk_storage"],
                "artifact": "docs/internal/agent-notes/current-lfm25-proof.json",
            }
        }
    }

    mixed_ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["mimo_v2", "dsv4"],
            "mixed_model_identity_families": ["minimax"],
            "resource_blockers": {
                "dsv4": {"artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT}
            },
        },
    )
    assert [blocker["id"] for blocker in mixed_ledger["blockers"]] == [
        "real_ui_mixed_model_identity_blocked",
        "real_ui_unblocked_non_mimo_missing",
    ]


def test_release_blocker_ledger_uses_step37_vlm_runtime_audit_for_missing_ui_family():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        step37_vlm_runtime_audit={
            "status": "open",
            "artifact": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
            "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
            "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
            "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
            "mlx_vlm_step3p7_importable": False,
            "mlx_vlm_step3p7_runtime_available": False,
            "mlx_vlm_step3p7_runtime": {
                "importable": False,
                "available": False,
            },
            "local_reference_vlm_runtime": {
                "runtime_kind": "torch_reference_not_mlx",
                "has_step3p7_model": True,
                "has_step3_vl_processor": True,
                "has_patch_image_inputs": True,
            },
            "step_jangtq_status": {
                "artifact_format": "jang",
                "quantization_method": "jang-importance",
                "quantization_profile": "JANG_2L",
                "quantization_backend": "mx.quantize",
                "step_jangtq_available": False,
                "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
            },
            "mlx_vlm_implementation_contract": {
                "closest_reference_packages": [
                    "mlx_vlm.models.qwen3_vl_moe",
                    "mlx_vlm.models.lfm2_vl",
                    "mlx_vlm.models.gemma4",
                ],
                "required_module_files": [
                    "__init__.py",
                    "config.py",
                    "language.py",
                    "step3p7.py",
                    "vision.py",
                    "processing_step3p7.py",
                ],
                "required_capabilities": [
                    "image_patch_processing",
                    "2d_vision_rope",
                    "full_and_sliding_attention_cache",
                    "head_wise_attention_gate",
                    "multimodal_embedding_merge",
                ],
                "explicitly_rejected_clearance_paths": [
                    "text_only_bridge",
                    "importable_stub",
                ],
            },
            "source_owned_runtime_progress": {
                "module": "vmlx_engine.models.step3p7_mlx_vlm",
                "config_layer_available": True,
                "model_layer_available": True,
                "vision_layer_available": True,
                "projector_layer_available": True,
                "processor_layer_available": True,
                "image_embeds_merge_available": True,
                "pixel_values_processor_available": True,
                "vision_patch_embed_available": True,
                "vision_abs_posemb_resize_available": True,
                "vision_downsamplers_available": True,
                "release_clearance": "source_runtime_surface_present_needs_live_proof",
            },
        },
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["step37"],
            "resource_blockers": {},
        },
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "real_ui_step37_vlm_runtime_missing",
                "status": "open",
                "evidence": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
                "next_proof": (
                    "Implement and prove a real Step3p7 VLM runtime path before "
                    "rerunning the Step3.7 real Electron UI image proof."
                ),
                "details": {
                    "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                    "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                    "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                    "mlx_vlm_step3p7_importable": False,
                    "mlx_vlm_step3p7_runtime_available": False,
                    "mlx_vlm_step3p7_runtime": {
                        "importable": False,
                        "available": False,
                    },
                    "local_reference_vlm_runtime": {
                        "runtime_kind": "torch_reference_not_mlx",
                        "has_step3p7_model": True,
                        "has_step3_vl_processor": True,
                        "has_patch_image_inputs": True,
                    },
                    "step_jangtq_status": {
                        "artifact_format": "jang",
                        "quantization_method": "jang-importance",
                        "quantization_profile": "JANG_2L",
                        "quantization_backend": "mx.quantize",
                        "step_jangtq_available": False,
                        "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
                    },
                    "mlx_vlm_implementation_contract": {
                        "closest_reference_packages": [
                            "mlx_vlm.models.qwen3_vl_moe",
                            "mlx_vlm.models.lfm2_vl",
                            "mlx_vlm.models.gemma4",
                        ],
                        "required_module_files": [
                            "__init__.py",
                            "config.py",
                            "language.py",
                            "step3p7.py",
                            "vision.py",
                            "processing_step3p7.py",
                        ],
                        "required_capabilities": [
                            "image_patch_processing",
                            "2d_vision_rope",
                            "full_and_sliding_attention_cache",
                            "head_wise_attention_gate",
                            "multimodal_embedding_merge",
                        ],
                        "explicitly_rejected_clearance_paths": [
                            "text_only_bridge",
                            "importable_stub",
                        ],
                    },
                    "source_owned_runtime_progress": {
                        "module": "vmlx_engine.models.step3p7_mlx_vlm",
                        "config_layer_available": True,
                        "model_layer_available": True,
                        "vision_layer_available": True,
                        "projector_layer_available": True,
                        "processor_layer_available": True,
                        "image_embeds_merge_available": True,
                        "pixel_values_processor_available": True,
                        "vision_patch_embed_available": True,
                        "vision_abs_posemb_resize_available": True,
                        "vision_downsamplers_available": True,
                        "release_clearance": "source_runtime_surface_present_needs_live_proof",
                    },
                },
            }
        ],
        "deferred_release_gaps": [],
        "deferred_release_families": [{"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"}],
    }


def test_release_blocker_ledger_keeps_step37_as_live_ui_gap_when_runtime_audit_passes():
    ledger = _current_release_blocker_ledger(
        regression_suite={"open_requirements": []},
        live_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        live_tool_smoke_summaries={"status": "pass", "missing": [], "not_pass": []},
        mimo_v2_jang2l_sink_ab={"status": "pass"},
        mimo_v2_jang2l_root_cause={"remote_evidence_only": False},
        issue175_179_release_boundary_audit={"status": "pass", "issues": {}},
        installed_app_runtime_parity_audit={"status": "pass"},
        issue179_minimax_k_root_cause_audit={"status": "pass"},
        step37_vlm_runtime_audit={
            "status": "pass",
            "artifact": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
            "root_cause": "step37_vlm_runtime_present_or_not_applicable",
            "release_clearance": "audit_does_not_block_release",
            "mlx_vlm_step3p7_runtime_available": True,
        },
        real_ui_live_model_matrix={
            "status": "open",
            "missing_families": ["step37"],
            "resource_blockers": {},
        },
    )

    assert ledger == {
        "status": "open",
        "blockers": [
            {
                "id": "real_ui_unblocked_non_mimo_missing",
                "status": "open",
                "evidence": (
                    "current_proof_sweep.real_ui_live_model_matrix."
                    "missing_families:step37"
                ),
                "next_proof": (
                    "Run real Electron UI proofs for the missing "
                    "non-MiMo, non-resource-blocked families on this "
                    "laptop before release."
                ),
            }
        ],
        "deferred_release_gaps": [],
        "deferred_release_families": [{"family": "dsv4", "reason": "deferred_per_20260602_emergency_release_scope"}],
    }


def test_release_manifest_rejects_step37_audit_without_reference_runtime_markers(tmp_path):
    artifact = tmp_path / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                "mlx_vlm_step3p7_runtime_available": False,
                "local_bridge_kind": "text_only",
                "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                "local_reference_vlm_runtime": {
                    "runtime_kind": "torch_reference_not_mlx",
                    "has_modeling_step3p7": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_step37_vlm_runtime_audit(tmp_path)

    assert result["status"] == "fail"
    assert (
        "missing_step37_reference_runtime_marker:has_2d_vision_rope"
        in result["failures"]
    )
    assert (
        "missing_step37_reference_runtime_marker:has_head_wise_attention_gate"
        in result["failures"]
    )


def test_release_manifest_rejects_step37_audit_without_jangtq_absence_marker(tmp_path):
    artifact = tmp_path / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                "mlx_vlm_step3p7_runtime_available": False,
                "local_bridge_kind": "text_only",
                "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                "local_reference_vlm_runtime": {
                    "runtime_kind": "torch_reference_not_mlx",
                    "has_modeling_step3p7": True,
                    "has_processing_step3": True,
                    "has_vision_encoder": True,
                    "has_configuration_step3p7": True,
                    "has_step3p7_model": True,
                    "has_conditional_generation": True,
                    "has_step3_vl_processor": True,
                    "has_multimodal_merge": True,
                    "has_patch_image_inputs": True,
                    "has_2d_vision_rope": True,
                    "has_full_and_sliding_attention_masks": True,
                    "has_qk_norms": True,
                    "has_head_wise_attention_gate": True,
                    "has_step3p7_config": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_step37_vlm_runtime_audit(tmp_path)

    assert result["status"] == "fail"
    assert "missing_step37_jangtq_absence_marker" in result["failures"]


def test_release_manifest_rejects_step37_audit_without_mlx_vlm_port_contract(tmp_path):
    artifact = tmp_path / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "open",
                "model_type": "step3p7",
                "text_model_type": "step3p5",
                "has_vision_config": True,
                "jang_has_vision": True,
                "root_cause": "missing_mlx_vlm_step3p7_vlm_runtime",
                "mlx_vlm_step3p7_runtime_available": False,
                "local_bridge_kind": "text_only",
                "release_clearance": "blocked_until_real_step3p7_vlm_runtime",
                "failure_summary": "No module named 'mlx_vlm.models.step3p7'",
                "local_reference_vlm_runtime": {
                    "runtime_kind": "torch_reference_not_mlx",
                    "has_modeling_step3p7": True,
                    "has_processing_step3": True,
                    "has_vision_encoder": True,
                    "has_configuration_step3p7": True,
                    "has_step3p7_model": True,
                    "has_conditional_generation": True,
                    "has_step3_vl_processor": True,
                    "has_multimodal_merge": True,
                    "has_patch_image_inputs": True,
                    "has_2d_vision_rope": True,
                    "has_full_and_sliding_attention_masks": True,
                    "has_qk_norms": True,
                    "has_head_wise_attention_gate": True,
                    "has_step3p7_config": True,
                },
                "step_jangtq_status": {
                    "quantization_profile": "JANG_2L",
                    "step_jangtq_available": False,
                    "step_jangtq_reason": "step_jangtq_not_made_current_artifact_is_jang_2l",
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validate_current_step37_vlm_runtime_audit(tmp_path)

    assert result["status"] == "fail"
    assert "missing_step37_mlx_vlm_implementation_contract" in result["failures"]


def test_current_proof_sweep_does_not_accept_mimo_as_current_open_gap():
    assert "mimo_v2_jang2l" not in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS
    assert not _live_smoke_gap_is_expected_mimo_open(
        {
            "status": "fail",
            "missing": [],
            "not_pass": [{"artifact": "build/current-mimo-stale.json", "status": "fail"}],
            "non_mimo_status": "pass",
            "non_mimo_missing": [],
            "non_mimo_not_pass": [],
        },
        "build/current-mimo-stale.json",
    )


def test_live_smoke_summary_tracks_non_mimo_status_separately(tmp_path):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert "mimo_v2_jang2l" not in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS
    assert result["status"] == "pass"
    assert result["non_mimo_status"] == "pass"
    assert result["non_mimo_missing"] == []
    assert result["non_mimo_not_pass"] == []


def test_live_tool_smoke_summary_tracks_non_mimo_status_separately(tmp_path):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_tool_smoke_artifacts,
    )

    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)

    result = _validate_current_covered_live_tool_smoke_artifacts(tmp_path)

    assert "mimo_v2_jang2l" not in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS
    assert result["status"] == "pass"
    assert result["non_mimo_status"] == "pass"
    assert result["non_mimo_missing"] == []
    assert result["non_mimo_not_pass"] == []


def test_release_regression_manifest_rejects_stale_current_objective_digest(tmp_path):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
                **_passing_current_suite_progress_checkpoints(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_current_objective_digest(
        tmp_path,
        open_requirements=[
            *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "Unexpected objective row is still open",
        ],
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["component_ok"]["objective_digest"] is False
    assert result["objective_digest"]["unexpected_open_requirements"] == [
        "Unexpected objective row is still open"
    ]


def test_release_regression_manifest_rejects_missing_or_failing_current_artifacts(tmp_path):
    artifacts = list(CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values())
    for artifact in artifacts[1:]:
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "fail" if artifact == artifacts[1] else "pass"
        path.write_text(f'{{"status":"{status}","failed":["example"]}}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
                **_passing_current_suite_progress_checkpoints(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["missing"] == [artifacts[0]]
    assert result["not_pass"] == [{"artifact": artifacts[1], "status": "fail"}]


def test_release_regression_manifest_rejects_stale_dsv4_open_requirement_details(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": {
                    "Real Electron UI cross-family live model matrix is release-cleared": _passing_open_requirement_details()[
                        "Real Electron UI cross-family live model matrix is release-cleared"
                    ],
                    "DSV4 long-output/code/file-generation quality is release-cleared": {
                        "details": {
                            "direct_off_exactness_boundary": {
                                "hidden_force_on_would_be_false_clearance": True,
                            },
                            "current_source_full_output_preflight": _passing_open_requirement_details()[
                                "DSV4 long-output/code/file-generation quality is release-cleared"
                            ]["details"]["current_source_full_output_preflight"],
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_exact_code_root_boundary",
        }
    ]


def test_release_regression_manifest_rejects_stale_issue179_objective_digest_row(
    tmp_path,
):
    _write_current_objective_digest(tmp_path)
    path = tmp_path / CURRENT_OBJECTIVE_DIGEST_ARTIFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    unexpected_requirement = "Unexpected MiniMax stale reporter row is release-cleared"
    payload["requirements"].append(
        {
            "requirement": unexpected_requirement,
            "status": "open",
            "details": {"evidence_files_present": True, "missing_evidence": []},
        }
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["objective_digest"]["unexpected_open_requirements"] == [unexpected_requirement]


def test_release_regression_manifest_rejects_missing_dsv4_source_preflight_boundary(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    assert isinstance(dsv4_row, dict)
    details = dsv4_row["details"]
    assert isinstance(details, dict)
    details.pop("current_source_full_output_preflight")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_narrow_dsv4_source_preflight_cases(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["selected_cases"] = ["chat_off_rep1"]
    source_preflight["case_count"] = 1
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_accepts_vm_stat_dsv4_source_preflight_reason(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["reason"] = "insufficient_vm_stat_memory"
    source_preflight["available_gb"] = 130.0
    source_preflight["available_for_gate_gb"] = 84.58
    source_preflight["free_plus_speculative_purgeable_gb"] = 84.58
    source_preflight["strict_vm_stat_memory_gap_gb"] = 35.42
    source_preflight["psutil_available_gap_gb"] = 0.0
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["regression_suite"]["open_requirement_detail_failures"] == []


def test_release_regression_manifest_accepts_current_memory_gated_dsv4_open_boundary(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    details = dsv4_row["details"]
    details["direct_off_exactness_boundary"] = {
        "present": True,
        "direct_off_release_blocker_active": True,
        "requested_thinking_success_is_diagnostic_only": True,
        "requested_thinking_success_clears_direct_off": False,
        "hidden_force_on_would_be_false_clearance": True,
        "all_direct_off_failed_routes": ["chat"],
        "requested_thinking_exact_routes": ["responses"],
    }
    details["exact_code_root_boundary"] = {
        "present": True,
        "current_primary_failure": "insufficient_evidence",
        "source_full_output_clearance_missing": True,
        "requested_thinking_is_diagnostic_only": True,
    }
    source_preflight = details["current_source_full_output_preflight"]
    source_preflight["reason"] = "insufficient_vm_stat_memory"
    source_preflight["model"] = "/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"
    source_preflight["available_gb"] = 115.76
    source_preflight["available_for_gate_gb"] = 111.7
    source_preflight["free_plus_speculative_purgeable_gb"] = 111.7
    source_preflight["strict_vm_stat_memory_gap_gb"] = 8.3
    source_preflight["psutil_available_gap_gb"] = 4.24
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["regression_suite"]["open_requirement_detail_failures"] == []


def test_release_regression_manifest_rejects_partial_four_case_dsv4_source_preflight(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["selected_cases"] = [
        "chat_off_rep1",
        "chat_off_no_punct_rep1",
        "responses_off_rep1",
        "responses_off_no_punct_rep1",
    ]
    source_preflight["case_count"] = 4
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_dsv4_source_preflight_without_no_launch_context(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["launch_allowed"] = True
    source_preflight["launch_decision"] = "launch_allowed"
    source_preflight["did_not_launch"] = False
    source_preflight["active_heavy_process_count"] = 1
    source_preflight["active_heavy_processes"] = [
        {"pid": 1234, "command": "vmlx_engine.cli serve --model dsv4"}
    ]
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_dsv4_source_preflight_without_vm_stat_memory_source(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["preflight_memory_source"] = "psutil_available"
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_stale_dsv4_source_preflight_artifact(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight["artifact"] = (
        "build/current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0625.json"
    )
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_undated_dsv4_source_preflight(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    open_details = _passing_open_requirement_details()
    dsv4_row = open_details[
        "DSV4 long-output/code/file-generation quality is release-cleared"
    ]
    source_preflight = dsv4_row["details"]["current_source_full_output_preflight"]
    source_preflight.pop("created_at", None)
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": open_details,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"]["open_requirement_detail_failures"] == [
        {
            "requirement": "DSV4 long-output/code/file-generation quality is release-cleared",
            "reason": "missing_or_stale_dsv4_source_full_output_preflight",
        }
    ]


def test_release_regression_manifest_rejects_missing_or_failing_covered_live_smoke_artifacts(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    missing_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    failing_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    (tmp_path / missing_artifact).unlink()
    row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS["qwen36_moe_crack"])
    failing_result = {
        "model": "Qwen3.6-27B-JANG_4M-MTP",
        "row": row,
        "status": "fail",
        "failures": ["unexpected_cjk_visible_text"],
    }
    failing_result["requests"] = [
        {"label": label, "validation_failures": []}
        for label in _required_live_smoke_request_labels(
            "qwen36_moe_crack",
            failing_result,
        )
    ]
    (tmp_path / failing_artifact).write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 1,
                "results": [failing_result],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["live_smoke_summaries"]["missing"] == [missing_artifact]
    not_pass = result["live_smoke_summaries"]["not_pass"]
    entry = next(item for item in not_pass if item["artifact"] == failing_artifact)
    failing = entry["failing_results"][0]
    assert entry["status"] == "pass"
    assert entry["failed"] == 1
    assert failing["model"] == "Qwen3.6-27B-JANG_4M-MTP"
    assert failing["status"] == "fail"
    assert failing["failures"] == ["unexpected_cjk_visible_text"]
    assert failing["missing_request_labels"] == []
    assert failing["request_validation_failures"] == []


def test_release_regression_manifest_rejects_live_smoke_top_level_fail_status(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload["status"] = "fail"
    payload["failed"] = 0
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "fail",
            "completed": 1,
            "failed": 0,
            "failing_results": [],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_missing_top_level_status(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload.pop("status")
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "missing_status",
            "completed": 1,
            "failed": 0,
            "failing_results": [],
        }
    ]


def test_release_regression_manifest_rejects_source_python_live_smoke_surface(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["ling_flash_tq"]
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"ling_flash_tq": artifact},
    )
    row = dict(CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS["ling_flash_tq"])
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "model": row["name"],
                        "row": row,
                        "status": "pass",
                        "failures": [],
                        "command": [
                            ".venv/bin/python",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            row["name"],
                        ],
                        "requests": [
                            {"label": label, "validation_failures": []}
                            for label in _required_live_smoke_request_labels(
                                "ling_flash_tq",
                                {"row": row},
                            )
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    entry = next(item for item in result["not_pass"] if item["artifact"] == artifact)
    failing = next(
        row
        for row in entry["failing_results"]
        if "surface_issues" in row
    )
    assert failing["surface_issues"] == [
        "command_python_not_bundled"
    ]


def test_release_regression_manifest_rejects_missing_or_wrong_tool_smoke_artifacts(
    tmp_path,
):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)

    missing_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["gemma4_crack"]
    failing_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    (tmp_path / missing_artifact).unlink()
    payload = json.loads((tmp_path / failing_artifact).read_text(encoding="utf-8"))
    tool_request = payload["results"][0]["requests"][-1]
    tool_request["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"value": "example"}
    )
    (tmp_path / failing_artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["live_tool_smoke_summaries"]["missing"] == [missing_artifact]
    assert result["live_tool_smoke_summaries"]["not_pass"] == [
        {
            "artifact": failing_artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-JANG_4M-MTP",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [
                        "tool_required:expected_tool_argument_missing"
                    ],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_tool_smoke_missing_top_level_status(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_tool_smoke_artifacts,
    )

    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["gemma4_crack"]
    payload = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    payload.pop("status")
    (tmp_path / artifact).write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_tool_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "missing_status",
            "completed": 1,
            "failed": 0,
            "failing_results": [],
        }
    ]


def test_release_regression_manifest_rejects_live_tool_smoke_without_repeat_cache_hit(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_tool_smoke_artifacts,
    )

    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS["gemma4_crack"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    for request in payload["results"][0]["requests"]:
        if request.get("label") == "text_cache_repeat_2":
            request["usage"] = {"prompt_tokens_details": None}
            request["cache_summary"] = {
                "has_cache_hit": False,
                "cache_hit_tokens": 0,
            }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_tool_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [
                        "text_cache_repeat_2:expected_cache_hit_missing"
                    ],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_missing_or_changed_diagnostic_smoke_artifacts(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS,
        _validate_current_diagnostic_live_smoke_artifacts,
    )

    _write_expected_diagnostic_live_smoke_artifacts(tmp_path)

    result = _validate_current_diagnostic_live_smoke_artifacts(tmp_path)

    assert CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS == {}
    assert result == {
        "status": "pass",
        "artifacts": {},
        "missing": [],
        "not_expected": [],
    }


def test_release_regression_manifest_tracks_zaya_vl_reasoning_media_diagnostic():
    assert CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS == {}


def test_release_regression_manifest_rejects_live_smoke_missing_required_request_coverage(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Qwen3.6-27B-JANG_4M-MTP",
                            "model_type": "qwen3_5",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Qwen3.6-27B-JANG_4M-MTP",
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-JANG_4M-MTP",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [
                        "reasoning_on",
                        "text_no_media_after_image",
                        "text_no_media_after_video",
                        "vl_blue_image",
                        "vl_blue_image_repeat",
                        "vl_blue_video",
                        "vl_red_image_changed",
                    ],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_without_repeat_cache_hit(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    for request in payload["results"][0]["requests"]:
        if request.get("label") == "text_cache_repeat_2":
            request["usage"] = {"prompt_tokens_details": None}
            request["cache_summary"] = {
                "has_cache_hit": False,
                "cache_hit_tokens": 0,
            }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [
                        "text_cache_repeat_2:expected_cache_hit_missing"
                    ],
                    "identity_mismatches": {},
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_video_capable_live_smoke_without_video_labels(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = "build/current-all-local-model-smoke-nemotron-omni-missing-video-test/summary.json"
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"nemotron_omni_tq2_system_nomedia": artifact},
    )
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                            "model_type": "nemotron_h",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        ],
                        "requests": [
                            {
                                "label": "text_cache_repeat_1",
                                "validation_failures": [],
                                "usage": {"prompt_tokens_details": None},
                                "cache_summary": {
                                    "has_cache_hit": False,
                                    "cache_hit_tokens": 0,
                                },
                            },
                            {
                                "label": "text_cache_repeat_2",
                                "validation_failures": [],
                                "usage": {
                                    "prompt_tokens_details": {
                                        "cached_tokens": 10,
                                        "cache_detail": "paged",
                                    }
                                },
                                "cache_summary": {
                                    "has_cache_hit": True,
                                    "cache_hit_tokens": 10,
                                },
                            },
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    entry = next(item for item in result["not_pass"] if item["artifact"] == artifact)
    failing = entry["failing_results"][0]
    assert entry["status"] == "pass"
    assert entry["completed"] == 1
    assert entry["failed"] == 0
    assert failing["model"] == "Nemotron-Omni-Nano-JANGTQ-CRACK"
    assert failing["missing_request_labels"] == [
        "text_no_media_after_video",
        "vl_blue_video",
    ]
    assert failing["request_validation_failures"] == []
    assert failing["identity_mismatches"] == {
        "supports_video": {
            "expected": False,
            "actual": True,
        },
    }


def test_release_regression_manifest_uses_effective_runtime_video_capability(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS[
        "nemotron_omni_tq2_system_nomedia"
    ]
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"nemotron_omni_tq2_system_nomedia": artifact},
    )
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                            "model_type": "nemotron_h",
                            "is_mllm": True,
                            "supports_video": False,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "capabilities": {
                            "body": {
                                "family": "nemotron_h",
                                "tool_parser": "nemotron",
                                "reasoning_parser": "deepseek_r1",
                                "modalities": ["text", "audio", "image"],
                            }
                        },
                        "probe_options": {
                            "include_media": True,
                            "include_video": False,
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Nemotron-Omni-Nano-JANGTQ-CRACK",
                        ],
                        "requests": [
                            {
                                "label": "text_cache_repeat_1",
                                "validation_failures": [],
                                "usage": {"prompt_tokens_details": None},
                                "cache_summary": {
                                    "has_cache_hit": False,
                                    "cache_hit_tokens": 0,
                                },
                            },
                            {
                                "label": "text_cache_repeat_2",
                                "validation_failures": [],
                                "usage": {
                                    "prompt_tokens_details": {
                                        "cached_tokens": 10,
                                        "cache_detail": "paged",
                                    }
                                },
                                "cache_summary": {
                                    "has_cache_hit": True,
                                    "cache_hit_tokens": 10,
                                },
                            },
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "pass"
    assert result["not_pass"] == []


def test_release_regression_manifest_does_not_treat_skipped_video_probe_as_capability_proof(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    artifact = "build/current-all-local-model-smoke-video-probe-skipped/summary.json"
    monkeypatch.setattr(
        manifest,
        "CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS",
        {"nemotron_omni_tq2_system_nomedia": artifact},
    )
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
                            "model_type": "nemotron_h",
                            "is_mllm": True,
                            "supports_video": True,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "probe_options": {
                            "include_media": True,
                            "include_video": False,
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Qwen3.6-27B-JANG_4M-MTP",
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"][0]["failing_results"][0]["missing_request_labels"] == [
        "text_no_media_after_video",
        "vl_blue_video",
    ]


def test_release_regression_manifest_rejects_live_smoke_wrong_model_identity(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["gemma4_crack"]
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "completed": 1,
                "failed": 0,
                "results": [
                    {
                        "row": {
                            "name": "Qwen3.6-27B-JANG_4M-MTP",
                            "model_type": "qwen3_5",
                            "is_mllm": True,
                            "supports_video": False,
                            "supports_thinking": True,
                            "cache_family": "hybrid_ssm",
                        },
                        "status": "pass",
                        "failures": [],
                        "command": [
                            "panel/bundled-python/python/bin/python3.12",
                            "-B",
                            "-s",
                            "-m",
                            "vmlx_engine.cli",
                            "serve",
                            "Qwen3.6-27B-JANG_4M-MTP",
                        ],
                        "requests": [
                            {"label": "text_cache_repeat_1", "validation_failures": []},
                            {"label": "text_cache_repeat_2", "validation_failures": []},
                            {"label": "text_multiturn_recall", "validation_failures": []},
                            {"label": "reasoning_on", "validation_failures": []},
                            {"label": "vl_blue_image", "validation_failures": []},
                            {"label": "text_no_media_after_image", "validation_failures": []},
                            {"label": "vl_blue_image_repeat", "validation_failures": []},
                            {"label": "vl_red_image_changed", "validation_failures": []},
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-JANG_4M-MTP",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {
                        "model_type": {
                            "expected": "gemma4",
                            "actual": "qwen3_5",
                        },
                        "name": {
                            "expected": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
                            "actual": "Qwen3.6-27B-JANG_4M-MTP",
                        },
                        "cache_family": {
                            "expected": "swa_rotating",
                            "actual": "hybrid_ssm",
                        },
                    },
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_wrong_capability_family(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["qwen36_moe_crack"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["capabilities"] = {
        "code": 200,
        "body": {
            "family": "gemma4",
            "tool_parser": "qwen",
            "reasoning_parser": None,
            "supports_thinking": True,
            "modalities": ["text", "vision"],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "Qwen3.6-27B-JANG_4M-MTP",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                    "capability_mismatches": {
                        "family": {
                            "expected": "qwen3_5",
                            "actual": "gemma4",
                        },
                        "reasoning_parser": {
                            "expected": "qwen3",
                            "actual": None,
                        },
                    },
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_live_smoke_wrong_runtime_parsers(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)
    artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS["dsv4_jang_local"]
    path = tmp_path / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["capabilities"] = {
        "code": 200,
        "body": {
            "family": "deepseek_v4",
            "tool_parser": "deepseek",
            "reasoning_parser": None,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["not_pass"] == [
        {
            "artifact": artifact,
            "status": "pass",
            "completed": 1,
            "failed": 0,
            "failing_results": [
                {
                    "model": "DeepSeek-V4-Flash-JANGTQ-K",
                    "status": "pass",
                    "failures": [],
                    "missing_request_labels": [],
                    "request_validation_failures": [],
                    "identity_mismatches": {},
                    "capability_mismatches": {
                        "tool_parser": {
                            "expected": "dsml",
                            "actual": "deepseek",
                        },
                        "reasoning_parser": {
                            "expected": "deepseek_r1",
                            "actual": None,
                        },
                    },
                }
            ],
        }
    ]


def test_release_regression_manifest_rejects_mimo_v2_stale_qwen_reasoning_parser(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_covered_live_smoke_artifacts,
    )

    _write_passing_covered_live_smoke_artifacts(tmp_path)

    result = _validate_current_covered_live_smoke_artifacts(tmp_path)

    assert "mimo_v2_jang2l" not in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS
    assert "mimo_v2_jang2l" not in result["artifacts"]
    assert result["status"] == "pass"


def test_release_regression_manifest_requires_mimo_v2_root_cause_artifacts(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "pass"
    assert result["metadata_truth_passed"] is True
    assert result["source_vs_quant_first_divergence_passed"] is True
    assert result["structural_verify_passed"] is True
    assert result["text_cache_narrow_pass"] is True
    assert result["switchglu_selected_expert_parity_passed"] is True
    assert result["prompt_length_coherence_blocked"] is False
    assert result["tool_protocol_blocked"] is False
    assert result["remote_evidence_only"] is False
    assert result["remote_artifacts"] == []
    assert result["local_release_clearance"] is True


def test_mimo_v2_root_cause_exposes_decode_speed_and_media_blockers(tmp_path):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    current_audit_path = tmp_path / CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    current_audit = json.loads(current_audit_path.read_text(encoding="utf-8"))
    current_audit["status"] = "open"
    current_audit["local_release_clearance"] = False
    current_audit["component_ok"]["decode_speed_target"] = False
    current_audit["component_ok"]["mimo_media_wired"] = False
    current_audit["component_ok"]["mimo_jangtq2_live_media_l2"] = True
    current_audit["component_ok"]["mimo_jang2l_live_media_l2"] = False
    current_audit["component_ok"]["mimo_jang2l_l2_restart_cache_hit"] = True
    current_audit["component_ok"]["mimo_jang2l_l2_restart_visible_output"] = False
    current_audit["blockers"] = [
        "mimo_decode_speed_below_release_target",
        "mimo_vl_audio_video_unwired",
        "mimo_jang2l_live_media_l2_missing",
        "mimo_jang2l_l2_restart_visible_output_blocked",
    ]
    current_audit["latest_decode_speed_evidence"] = {
        "status": "review",
        "speed_blocked": True,
        "bundle_decode_tps": 1.79,
        "greedy_decode_tps": 1.83,
        "coherency_exact": True,
        "native_cache_type": "mixed_swa_kv",
        "generic_turboquant_kv_enabled": False,
        "switchglu_fastpath_active": True,
        "switchglu_fastpath_max_calls": 4096,
        "switchglu_fastpath_max_compiled_shapes": 2,
        "max_decode_next_last_async_ms": 597.49,
        "async_decode_wait_dominates": True,
        "decode_bottleneck_classification": (
            "switchglu_fastpath_active_but_metal_async_wait_dominates"
        ),
    }
    current_audit_path.write_text(json.dumps(current_audit) + "\n", encoding="utf-8")

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "open"
    assert result["decode_speed_target_blocked"] is True
    assert result["media_unwired"] is True
    assert result["mimo_jangtq2_live_media_l2_passed"] is True
    assert result["mimo_jang2l_live_media_l2_blocked"] is True
    assert result["mimo_jang2l_l2_restart_cache_hit_passed"] is True
    assert result["mimo_jang2l_l2_restart_visible_output_blocked"] is False
    assert result["latest_decode_speed_evidence"]["bundle_decode_tps"] == 1.79
    assert result["switchglu_fastpath_active_but_slow"] is True
    assert result["async_decode_wait_dominates"] is True
    assert (
        result["decode_bottleneck_classification"]
        == "switchglu_fastpath_active_but_metal_async_wait_dominates"
    )
    assert "mimo_decode_speed_below_release_target" in result["failures"]
    assert "mimo_media_unwired" in result["failures"]
    assert "mimo_jang2l_live_media_l2_missing" in result["failures"]
    assert "mimo_jang2l_l2_restart_visible_output_blocked" not in result["failures"]


def test_mimo_v2_root_cause_consumes_current_jang2l_l2_and_responses_tool_proofs(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    restart_path = tmp_path / CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    restart_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "classification": {
                            "status": "pass",
                            "cache_status": "pass",
                            "output_status": "review",
                            "checks": {
                                "restart_l2_restore_observed": True,
                                "second_cached_tokens_positive": True,
                                "block_disk_hit_positive": True,
                            },
                        }
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    responses_path = tmp_path / CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.write_text(
        json.dumps(
            {
                "status": "fail",
                "proven": {
                    "responses_endpoint_used": True,
                    "responses_delta_streaming": True,
                    "responses_previous_response_id_tool_followup": True,
                    "cache": {
                        "cache_hit_tokens": 4552,
                        "l2_block_tokens_on_disk": 4960,
                    },
                },
                "red": {
                    "long_tool_loop": False,
                    "semantic_drift": [
                        "REAL_UI_LIVE_TOOL_ONE drifted to REAL_UI_LAND_TOOL_ONE"
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_audit_path = tmp_path / CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    current_audit = json.loads(current_audit_path.read_text(encoding="utf-8"))
    current_audit["status"] = "open"
    current_audit["local_release_clearance"] = False
    current_audit["component_ok"]["mimo_media_wired"] = False
    current_audit["component_ok"]["mimo_jang2l_live_media_l2"] = False
    current_audit["component_ok"]["mimo_jang2l_l2_restart_cache_hit"] = False
    current_audit["component_ok"]["mimo_jang2l_l2_restart_visible_output"] = False
    current_audit_path.write_text(json.dumps(current_audit) + "\n", encoding="utf-8")

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "open"
    assert result["mimo_jang2l_l2_restart_cache_hit_passed"] is True
    assert result["mimo_jang2l_l2_restart_visible_output_blocked"] is False
    assert "mimo_jang2l_l2_restart_visible_output_blocked" not in result["failures"]
    assert result["mimo_jang2l_responses_transport_passed"] is True
    assert result["mimo_jang2l_responses_tool_semantics_blocked"] is True
    assert "mimo_jang2l_responses_tool_semantics_blocked" in result["failures"]
    assert "JANG_2L Responses/tool semantic drift" in result["release_boundary"]


def test_mimo_v2_root_cause_preserves_current_long_prompt_oom_blocker(tmp_path):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    length_path = tmp_path / CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT
    length_path.parent.mkdir(parents=True, exist_ok=True)
    length_path.write_text(
        json.dumps(
            {
                "status": "open",
                "classification": (
                    "intrinsic_python_mllm_long_prompt_metal_oom_first_request"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_audit_path = tmp_path / CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    current_audit = json.loads(current_audit_path.read_text(encoding="utf-8"))
    current_audit["status"] = "open"
    current_audit["local_release_clearance"] = False
    current_audit["component_ok"]["long_prompt_coherence"] = True
    current_audit_path.write_text(json.dumps(current_audit) + "\n", encoding="utf-8")

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "open"
    assert result["prompt_length_coherence_blocked"] is True
    assert (
        result["prompt_length_runtime_blocker"]
        == "intrinsic_python_mllm_long_prompt_metal_oom_first_request"
    )
    assert "long-prompt coherence" in result["release_boundary"]


def test_mimo_v2_root_cause_accepts_policy_skipped_source_vs_quant_without_clearing_quality(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    source_vs_quant_path = tmp_path / CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT
    source_vs_quant_path.write_text(
        json.dumps(
            {
                "status": "missing_prerequisites",
                "remote_evidence_only": False,
                "source_model_path": "/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5",
                "quant_model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
                "rows": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    classifier_path = (
        tmp_path / CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT
    )
    classifier_path.write_text(
        json.dumps(
            {
                "status": "open",
                "classification": "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes",
                "secondary_classification": "jang2l_json_sentinel_semantic_mismatch_open",
                "source_vs_quant_load_performed": False,
                "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
                "excluded_surfaces": {
                    "parser_argument_rewrite": True,
                    "prefix_paged_l2_or_kv_quant_primary_cause": True,
                    "hidden_stochastic_sampling_primary_cause": True,
                    "api_sampler_non_top1_selection": True,
                },
                "unresolved_surfaces": {
                    "artifact_quantization_or_decode_logits_quality": True,
                    "source_vs_quant_first_divergence": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_audit_path = tmp_path / CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    current_audit = json.loads(current_audit_path.read_text(encoding="utf-8"))
    current_audit["status"] = "open"
    current_audit["local_release_clearance"] = False
    current_audit["component_ok"]["source_vs_quant_first_divergence"] = False
    current_audit["component_ok"]["artifact_exactness"] = False
    current_audit["blockers"] = [
        "mimo_jangtq2_artifact_exactness_blocked",
        "mimo_source_vs_quant_first_divergence_missing_or_failed",
    ]
    current_audit_path.write_text(json.dumps(current_audit) + "\n", encoding="utf-8")

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "open"
    assert result["local_release_clearance"] is False
    assert result["source_vs_quant_first_divergence_passed"] is False
    assert result["source_vs_quant_first_divergence_policy_skipped"] is True
    assert result["source_vs_quant_requirement_satisfied"] is True
    assert (
        result["source_vs_quant_policy_skip_reason"]
        == "user_disallowed_source_vs_quant_due_ram"
    )
    assert "mimo_source_vs_quant_first_divergence_missing" not in result["failures"]
    assert (
        "mimo_source_vs_quant_first_divergence_missing_or_failed"
        not in result["current_audit_blockers"]
    )
    assert result["artifact_exactness_bundle_status"]["jangtq2"] == {
        "blocked": True,
        "classification": "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes",
        "evidence": CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
    }
    assert result["artifact_exactness_bundle_status"]["jang2l"] == {
        "blocked": True,
        "classification": "jang2l_json_sentinel_semantic_mismatch_open",
        "evidence": CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
    }
    assert "mimo_jangtq2_artifact_exactness_blocked" in result["failures"]
    assert "mimo_jang2l_artifact_exactness_blocked" in result["failures"]
    assert (
        "mimo_no_source_exactness_classifier_missing_literal_mutation_boundary"
        not in result["failures"]
    )
    assert "media wiring" not in result["release_boundary"]
    assert "decode speed" not in result["release_boundary"]
    assert "tool protocol" not in result["release_boundary"]
    assert "long-prompt coherence" not in result["release_boundary"]
    assert "current JANGTQ2 artifact exactness" in result["release_boundary"]
    assert "current JANG_2L artifact exactness" in result["release_boundary"]
    assert result["artifact_exactness_release_action"] == (
        "replace_all_routed_2bit_jangtq2_or_lift_gate_down_precision"
    )
    assert result["recommended_next_artifact_profiles"] == [
        "JANGTQ gate=3/up=2/down=3",
        "JANGTQ gate=3/up=3/down=3",
    ]


def test_mimo_v2_root_cause_maps_compact_hyphen_exactness_to_rebuild_action(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT,
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    _write_passing_mimo_v2_root_cause_artifacts(tmp_path)
    source_vs_quant_path = tmp_path / CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT
    source_vs_quant_path.write_text(
        json.dumps(
            {
                "status": "missing_prerequisites",
                "remote_evidence_only": False,
                "source_model_path": "/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5",
                "quant_model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2",
                "rows": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    classifier_path = (
        tmp_path / CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT
    )
    classifier_path.write_text(
        json.dumps(
            {
                "status": "open",
                "classification": "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer",
                "secondary_classification": "jang2l_json_sentinel_semantic_mismatch_open",
                "source_vs_quant_load_performed": False,
                "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
                "excluded_surfaces": {
                    "chat_template_only_primary_cause": True,
                    "parser_argument_rewrite": True,
                    "prefix_paged_l2_or_kv_quant_primary_cause": True,
                    "tokenizer_roundtrip_primary_cause": True,
                },
                "unresolved_surfaces": {
                    "artifact_quantization_or_decode_logits_quality": True,
                    "jangtq2_compact_hyphen_decode_quality": True,
                    "source_vs_quant_first_divergence": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_audit_path = tmp_path / CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    current_audit = json.loads(current_audit_path.read_text(encoding="utf-8"))
    current_audit["status"] = "open"
    current_audit["local_release_clearance"] = False
    current_audit["component_ok"]["source_vs_quant_first_divergence"] = False
    current_audit["component_ok"]["artifact_exactness"] = False
    current_audit["blockers"] = [
        "mimo_jangtq2_artifact_exactness_blocked",
        "mimo_source_vs_quant_first_divergence_missing_or_failed",
    ]
    current_audit_path.write_text(json.dumps(current_audit) + "\n", encoding="utf-8")

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["artifact_exactness_bundle_status"]["jangtq2"] == {
        "blocked": True,
        "classification": "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer",
        "evidence": CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
    }
    assert result["artifact_exactness_release_action"] == (
        "replace_all_routed_2bit_jangtq2_or_lift_gate_down_precision"
    )
    assert result["recommended_next_artifact_profiles"] == [
        "JANGTQ gate=3/up=2/down=3",
        "JANGTQ gate=3/up=3/down=3",
    ]


def test_mimo_v2_current_audit_extracts_fastpath_async_bottleneck(tmp_path):
    from tests.cross_matrix.run_mimo_v2_jang2l_current_audit import (
        _latest_decode_speed_evidence,
    )

    log_path = tmp_path / "mimo-speed.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO:vmlx_engine.models.mllm:MiMo-V2 affine SwitchGLU decode fast path active: calls=4096 compiled_shapes=2",
                "INFO:vmlx_engine.mllm_batch_generator:VMLINUX_DECODE_TRACE mllm steps=96 avg_model_ms=2.29 avg_sample_ms=0.38 last_model_ms=2.32 last_sample_ms=0.51 batch=1",
                "INFO:vmlx_engine.mllm_batch_generator:VMLINUX_DECODE_TRACE_NEXT mllm steps=96 last_total_ms=506.89 last_step_ms=2.93 last_async_ms=503.95 last_materialize_ms=0.01 prompt_processing=False batch=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = _latest_decode_speed_evidence(
        {
            "results": [
                {
                    "status": "review",
                    "log_path": str(log_path),
                    "bundle_sampling": {"decode_tps_wall": 1.79},
                    "greedy_topk0": {"decode_tps_wall": 1.83},
                    "coherency": {
                        "content_head": "READY\n17+28=45\nCERULEAN",
                        "loopish": False,
                    },
                    "health_after": {
                        "native_cache": {
                            "cache_type": "mixed_swa_kv",
                            "generic_turboquant_kv": {"enabled": False},
                        }
                    },
                }
            ]
        }
    )

    assert evidence["speed_blocked"] is True
    assert evidence["switchglu_fastpath_active"] is True
    assert evidence["switchglu_fastpath_max_calls"] == 4096
    assert evidence["switchglu_fastpath_max_compiled_shapes"] == 2
    assert evidence["max_decode_next_last_async_ms"] == 503.95
    assert evidence["async_decode_wait_dominates"] is True
    assert (
        evidence["decode_bottleneck_classification"]
        == "switchglu_fastpath_active_but_metal_async_wait_dominates"
    )


def test_mimo_v2_current_audit_accepts_packaged_jangtq2_decode_speed(tmp_path):
    from tests.cross_matrix.run_mimo_v2_jang2l_current_audit import (
        _latest_decode_speed_evidence,
    )

    log_path = tmp_path / "mimo-packaged-speed.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO:vmlx_engine.models.mllm:MiMo-V2 TurboQuant SwitchGLU decode fast path active: calls=4096 compiled_shapes=1",
                "INFO:vmlx_engine.server:Chat completion: 96 tokens in 2.39s (40.2 tok/s)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = _latest_decode_speed_evidence(
        {
            "results": [
                {
                    "status": "review",
                    "log_path": str(log_path),
                    "bundle_sampling": {"decode_tps_wall": 40.15},
                    "greedy_topk0": {"decode_tps_wall": 40.66},
                    "coherency": {
                        "content_head": "READY\n17+28=45\nCERULEAN",
                        "loopish": False,
                    },
                    "pp_rows": [{"pp_wall_tok_s": 127.76}],
                    "health_after": {
                        "native_cache": {
                            "cache_type": "mixed_swa_kv",
                            "generic_turboquant_kv": {"enabled": False},
                        }
                    },
                    "notes": ["PP below expected 400.00: 127.76"],
                }
            ]
        }
    )

    assert evidence["speed_blocked"] is False
    assert evidence["switchglu_fastpath_active"] is True
    assert evidence["switchglu_fastpath_max_calls"] == 4096
    assert evidence["switchglu_fastpath_max_compiled_shapes"] == 1
    assert evidence["pp_wall_tok_s"] == [127.76]
    assert (
        evidence["decode_bottleneck_classification"]
        == "switchglu_fastpath_active_without_async_trace_blocker"
    )


def test_current_proof_sweep_includes_mimo_root_cause_artifacts():
    from tests.cross_matrix.release_regression_manifest import (
        validate_current_proof_sweep_artifacts,
    )

    result = validate_current_proof_sweep_artifacts(Path.cwd())

    assert "mimo_v2_jang2l_root_cause" in result
    assert "mimo_v2_jang2l_root_cause" in result["component_ok"]


def test_current_mimo_v2_proof_artifact_constants_are_local_only():
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
    )

    artifacts = [
        CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT,
    ]

    assert all(artifact.startswith("build/current-") for artifact in artifacts)
    assert not any("/remote-" in artifact for artifact in artifacts)


def test_current_mimo_v2_proof_artifact_constants_use_latest_pointer_set():
    from tests.cross_matrix.release_regression_manifest import (
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
    )

    assert CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT == (
        "build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json"
    )
    assert CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT == (
        "build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json"
    )


def test_current_proof_sweep_tracks_mimo_root_cause_component():
    from tests.cross_matrix.release_regression_manifest import (
        validate_current_proof_sweep_artifacts,
    )

    result = validate_current_proof_sweep_artifacts(Path.cwd())

    assert "mimo_v2_jang2l_root_cause" in result
    assert "mimo_v2_jang2l_root_cause" in result["component_ok"]


def test_release_regression_manifest_rejects_missing_mimo_v2_root_cause_artifacts(
    tmp_path,
):
    from tests.cross_matrix.release_regression_manifest import (
        _validate_current_mimo_v2_jang2l_root_cause,
    )

    result = _validate_current_mimo_v2_jang2l_root_cause(tmp_path)

    assert result["status"] == "missing"
    assert result["missing"] == [
        CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT,
        CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT,
    ]


def test_release_regression_manifest_rejects_unexpected_current_regression_suite_state(tmp_path):
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": ["new_failure"],
                "open_requirements": [
                    *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                    "Server max output/context wiring regressed",
                ],
                "open_requirement_details": _passing_open_requirement_details(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["regression_suite"] == {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "pass",
        "failed_steps": ["new_failure"],
        "open_requirements": [
            *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "Server max output/context wiring regressed",
        ],
        "open_requirement_details": _passing_open_requirement_details(),
        "unexpected_open_requirements": ["Server max output/context wiring regressed"],
        "missing_expected_open_requirements": [],
        "open_requirement_detail_failures": [],
        "progress_checkpoint_failures": [
            "missing_current_step",
            "missing_completed_steps",
        ],
    }


def test_release_regression_manifest_rejects_stale_current_suite_source_hash(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    source_rel = "tests/cross_matrix/summarize_objective_proof.py"
    source_path = tmp_path / source_rel
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("new source\n", encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "source_hashes": {
                    source_rel: hashlib.sha256(b"old source\n").hexdigest()
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest, "CURRENT_SUITE_SOURCE_HASH_FILES", (source_rel,))

    result = manifest._validate_current_regression_suite_artifact(tmp_path)

    assert result["stale_source_hashes"] == [
        {
            "path": source_rel,
            "artifact_sha256": hashlib.sha256(b"old source\n").hexdigest(),
            "current_sha256": hashlib.sha256(b"new source\n").hexdigest(),
        }
    ]
    assert manifest._current_regression_suite_state_is_acceptable(result) is False


def test_release_regression_manifest_rejects_missing_current_suite_source_hashes(
    tmp_path,
    monkeypatch,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    source_rel = "tests/cross_matrix/summarize_objective_proof.py"
    source_path = tmp_path / source_rel
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("current source\n", encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "source_hashes": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manifest, "CURRENT_SUITE_SOURCE_HASH_FILES", (source_rel,))

    result = manifest._validate_current_regression_suite_artifact(tmp_path)

    assert result["missing_source_hashes"] == [source_rel]
    assert manifest._current_regression_suite_state_is_acceptable(result) is False


def test_release_regression_manifest_rejects_current_suite_without_progress_checkpoints(
    tmp_path,
):
    from tests.cross_matrix import release_regression_manifest as manifest

    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "open",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
                "steps": {
                    "objective_digest": {"status": "pass"},
                    "release_regression_manifest": {"status": "pass"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = manifest._validate_current_regression_suite_artifact(tmp_path)

    assert result["progress_checkpoint_failures"] == [
        "missing_current_step",
        "missing_completed_steps",
    ]
    assert manifest._current_regression_suite_state_is_acceptable(result) is False


def test_release_regression_manifest_requires_dsv4_generation_boundary_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "vmlx_engine/scheduler.py",
        "vmlx_engine/utils/dsv4_batch_generator.py",
        "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py",
        "tests/cross_matrix/run_dsv4_default_cache_tool_loop_gate.py",
        "tests/cross_matrix/run_gemma4_12b_speed_gate.py",
        "tests/test_dsv4_route_mode_code_exactness.py",
        "tests/test_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_gemma4_12b_speed_gate.py",
        "tests/test_objective_proof_digest.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))


def test_release_regression_manifest_source_hash_list_matches_current_suite_runner():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_current_regression_suite as suite

    assert manifest.CURRENT_SUITE_SOURCE_HASH_FILES == suite.CURRENT_SUITE_SOURCE_HASH_FILES


def test_release_regression_manifest_runner_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_release_regression_manifest as runner

    assert runner.DEFAULT_OUT == Path(
        "build/current-release-regression-manifest-after-pr-intake-matrix-refresh-20260609.json"
    )


def test_release_regression_manifest_requires_model_matrix_contract_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/cross_matrix/run_cache_architecture_contract.py",
        "tests/cross_matrix/run_generation_defaults_contract.py",
        "tests/cross_matrix/run_jang_model_compat_contract.py",
        "tests/cross_matrix/run_mcp_policy_contract.py",
        "tests/cross_matrix/run_model_family_detection_contract.py",
        "tests/cross_matrix/run_model_artifact_format_contract.py",
        "tests/cross_matrix/run_native_mtp_contract.py",
        "tests/cross_matrix/run_noheavy_api_cache_contract.py",
        "tests/cross_matrix/run_parser_registry_contract.py",
        "tests/cross_matrix/run_panel_tool_security_contract.py",
        "tests/cross_matrix/run_production_family_audit.py",
        "tests/cross_matrix/run_reasoning_template_contract.py",
        "tests/cross_matrix/run_release_regression_manifest.py",
        "tests/cross_matrix/run_release_surface_contract.py",
        "tests/cross_matrix/run_tool_call_contract.py",
        "tests/cross_matrix/run_vl_media_cache_contract.py",
        "tests/test_all_local_model_smoke.py",
        "vmlx_engine/api/tool_calling.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))


def test_release_regression_manifest_requires_release_blocker_boundary_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/cross_matrix/run_installed_app_runtime_parity_audit.py",
        "tests/cross_matrix/run_issue175_177_installed_runtime_audit.py",
        "tests/cross_matrix/run_issue175_177_live_runtime_audit.py",
        "tests/cross_matrix/run_issue175_179_release_boundary_audit.py",
        "tests/cross_matrix/run_issue181_183_runtime_audit.py",
        "tests/cross_matrix/run_public_app_issue_audit.py",
        "tests/cross_matrix/run_issue179_minimax_k_model_manifest.py",
        "tests/cross_matrix/run_issue179_reporter_parity_metadata.py",
        "tests/cross_matrix/run_issue179_minimax_k_root_cause_audit.py",
        "tests/cross_matrix/run_issue179_public_dmg_contract.py",
        "tests/cross_matrix/run_issue179_responses_cancel_probe.py",
        "tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py",
        "tests/test_installed_app_runtime_parity_audit.py",
        "tests/test_issue175_177_installed_runtime_audit.py",
        "tests/test_issue175_177_live_runtime_audit.py",
        "tests/test_issue175_179_release_boundary_audit.py",
        "tests/test_issue181_183_runtime_audit.py",
        "tests/test_public_app_issue_audit.py",
        "tests/test_issue179_minimax_k_model_manifest.py",
        "tests/test_issue179_reporter_parity_metadata.py",
        "tests/test_issue179_minimax_k_root_cause_audit.py",
        "tests/test_issue179_responses_cancel_probe.py",
        "tests/test_cancellation.py",
        "tests/test_real_ui_dsv4_memory_preflight.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_source_hashes_all_referenced_code_files():
    from tests.cross_matrix import release_regression_manifest as manifest

    text_parts: list[str] = []
    for row in manifest.build_manifest()["rows"]:
        for key in ("commands", "artifacts", "proves"):
            values = row.get(key, [])
            if isinstance(values, list):
                text_parts.extend(str(value) for value in values)
    joined = "\n".join(text_parts)
    referenced_paths = {
        match.rstrip(".,;)")
        for match in re.findall(
            r"((?:tests|panel|bench|vmlx_engine)/[^\s`\"']+)",
            joined,
        )
    }
    referenced_code = {
        path
        for path in referenced_paths
        if path.endswith((".py", ".mjs", ".ts", ".tsx", ".sh"))
        and Path(path).is_file()
    }

    missing = sorted(referenced_code - set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))

    assert missing == []


def test_release_regression_manifest_requires_runtime_launch_and_mllm_cache_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "bench/native_mtp_speed_ab.py",
        "tests/cross_matrix/run_decode_speed_gate.py",
        "vmlx_engine/block_disk_store.py",
        "vmlx_engine/cli.py",
        "vmlx_engine/engine/batched.py",
        "vmlx_engine/engine/simple.py",
        "vmlx_engine/mllm_scheduler.py",
        "vmlx_engine/models/mllm.py",
        "vmlx_engine/paged_cache.py",
        "vmlx_engine/reranker.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_requires_panel_api_settings_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "panel/src/main/api-gateway.ts",
        "panel/src/main/engine-manager.ts",
        "panel/scripts/live-chat-tools-reasoning-proof.mjs",
        "panel/src/main/index.ts",
        "panel/src/main/server.ts",
        "panel/src/main/sessions.ts",
        "panel/src/main/ipc/chat.ts",
        "panel/src/main/ipc/sessions.ts",
        "panel/src/main/ipc/developer.ts",
        "panel/src/main/ipc/image.ts",
        "panel/src/main/ipc/imageGenerationState.ts",
        "panel/src/main/ipc/models.ts",
        "panel/src/main/model-config-registry.ts",
        "panel/src/main/process-manager.ts",
        "panel/src/main/tools/executor.ts",
        "panel/src/renderer/src/components/chat/MessageBubble.tsx",
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
        "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/tests/api-gateway-ollama-behavior.test.ts",
        "panel/tests/api-gateway-ollama.test.ts",
        "panel/tests/api-gateway-single-model.behavior.test.ts",
        "panel/tests/generation-defaults.test.ts",
        "panel/tests/image-system.test.ts",
        "panel/tests/interleaved-reasoning-render.test.ts",
        "panel/tests/model-config-registry.test.ts",
        "panel/tests/settings-flow.test.ts",
        "tests/cross_matrix/run_api_surface_contract.py",
        "tests/cross_matrix/run_max_output_context_contract.py",
        "tests/cross_matrix/run_noheavy_panel_settings_contract.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_requires_focused_pytest_gate_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/test_objective_proof_digest.py",
        "tests/test_agents_release_control_plane.py",
        "tests/test_mimo_v2_no_source_exactness_classifier.py",
        "tests/test_dsv4_default_cache_tool_loop_gate.py",
        "tests/test_release_gate_python_app.py",
        "tests/test_current_regression_suite.py",
        "tests/test_release_regression_manifest.py",
        "tests/test_model_family_detection_contract.py",
        "tests/test_mcp_policy_contract.py",
        "tests/test_vl_media_cache_contract.py",
        "tests/test_batching.py",
        "tests/test_dsv4_batch_generator_speed.py",
        "tests/test_scheduler_repetition_context.py",
        "tests/test_dsv4_paged_cache.py",
    }

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_requires_dirty_contract_unit_source_hashes():
    from tests.cross_matrix import release_regression_manifest as manifest

    required = {
        "tests/test_api_surface_contract.py",
        "tests/test_cache_architecture_contract.py",
        "tests/test_dsml_tool_parser.py",
        "tests/test_engine_audit.py",
        "tests/test_generation_defaults_contract.py",
        "tests/test_image_api.py",
        "tests/test_image_gen.py",
        "tests/test_local_generation_metadata_audit.py",
        "tests/test_mllm_continuous_batching.py",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_model_config_registry.py",
        "tests/test_mlx_memory_cleanup.py",
        "tests/test_packaged_integrity_contract.py",
        "tests/test_paged_cache_unit.py",
        "tests/test_panel_cli_flag_contract.py",
        "tests/test_reasoning_modes.py",
        "tests/test_runtime_memory_stress_probe.py",
        "tests/test_server.py",
        "tests/test_tool_format.py",
        "tests/test_tool_parsers.py",
        "vmlx_engine/image_gen.py",
        "vmlx_engine/model_config_registry.py",
        "vmlx_engine/model_configs.py",
        "vmlx_engine/mlx_memory.py",
        "vmlx_engine/reasoning/__init__.py",
        "vmlx_engine/tool_parsers/__init__.py",
        "vmlx_engine/tool_parsers/xml_function_tool_parser.py",
        "vmlx_engine/tool_parsers/zaya_tool_parser.py",
    }
    required |= {str(path) for path in Path("vmlx_engine/reasoning").glob("*.py")}

    assert required.issubset(set(manifest.CURRENT_SUITE_SOURCE_HASH_FILES))
    assert all(Path(path).exists() for path in required)


def test_release_regression_manifest_allows_current_suite_bootstrap_self_reference(tmp_path):
    result = {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "open",
        "failed_steps": ["release_regression_manifest"],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
    }

    assert _current_regression_suite_state_is_acceptable(result)


def test_release_regression_manifest_allows_current_suite_open_with_only_expected_requirements(
    tmp_path,
):
    result = {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "open",
        "failed_steps": [],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "open_requirement_detail_failures": [],
    }

    assert _current_regression_suite_state_is_acceptable(result)


def test_release_regression_manifest_rejects_incomplete_current_max_output_context_matrix(tmp_path):
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == max_output_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "server startup maxTokens and chat maxTokens remain independent when both are set"
                        ],
                        "failed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["max_output_context_matrix"]["artifact"] == max_output_artifact
    assert result["max_output_context_matrix"]["status"] == "pass"
    assert result["max_output_context_matrix"]["missing_markers"] == [
        "server startup maxTokens and chat maxTokens remain independent when both are set"
    ]
    assert result["max_output_context_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_packaged_integrity_matrix(tmp_path):
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == packaged_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "failed": ["bundled_python_verifier"],
                        "known_expected_release_gate_open_requirements": [
                            *EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                            "Unexpected release gate blocker",
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["packaged_integrity_matrix"]["artifact"] == packaged_artifact
    assert result["packaged_integrity_matrix"]["status"] == "pass"
    assert result["packaged_integrity_matrix"]["failed"] == ["bundled_python_verifier"]
    assert result["packaged_integrity_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS[-1]
    ]
    assert result["packaged_integrity_matrix"]["unexpected_open_requirements"] == [
        "Unexpected release gate blocker"
    ]
    assert result["packaged_integrity_matrix"]["missing_expected_open_requirements"] == []


def test_release_regression_manifest_rejects_incomplete_current_release_surface_matrix(tmp_path):
    assert "staged_source_version_not_public" in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS

    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == release_surface_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["release_surface_matrix"]["artifact"] == release_surface_artifact
    assert result["release_surface_matrix"]["status"] == "pass"
    assert result["release_surface_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS[-1]
    ]


def test_release_regression_manifest_accepts_informational_release_surface_false_checks(tmp_path):
    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "public-release-surface-preflight"
    ]
    path = tmp_path / release_surface_artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS}
    checks["staged_source_version_not_public"] = False
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": checks,
                "status_failed_checks": [],
                "informational_false_checks": ["staged_source_version_not_public"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _validate_current_release_surface_matrix_artifact(tmp_path)

    assert result["status"] == "pass"
    assert result["failed_checks"] == []
    assert result["missing_expected_checks"] == []


def test_release_regression_manifest_rejects_incomplete_current_model_family_matrix(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == model_family_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS[:-1],
                        "missing_rows": [EXPECTED_CURRENT_MODEL_FAMILY_ROWS[-1]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["model_family_matrix"]["artifact"] == model_family_artifact
    assert result["model_family_matrix"]["status"] == "pass"
    assert result["model_family_matrix"]["missing_rows"] == [EXPECTED_CURRENT_MODEL_FAMILY_ROWS[-1]]
    assert result["model_family_matrix"]["missing_expected_rows"] == [EXPECTED_CURRENT_MODEL_FAMILY_ROWS[-1]]


def test_release_regression_manifest_rejects_incomplete_current_model_artifact_matrix(tmp_path):
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == model_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": ["test_native_mtp_detection_uses_weights_not_path_name"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["model_artifact_matrix"]["artifact"] == model_artifact
    assert result["model_artifact_matrix"]["status"] == "pass"
    assert result["model_artifact_matrix"]["missing_markers"] == [
        "test_native_mtp_detection_uses_weights_not_path_name"
    ]
    assert result["model_artifact_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_cache_architecture_matrix(tmp_path):
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": ["test_mllm_stats_include_cache_fields"],
                        "missing_api_checks": ["turboquant_disk_roundtrip"],
                        "missing_api_command_markers": ["generic_turboquant_patcher_skips_hybrid_ssm"],
                        "missing_panel_markers": ["deepseek-v4 disables composite prefix cache by default even with stale cache config"],
                        "cache_family_matrix": _passing_cache_family_matrix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["cache_architecture_matrix"]["artifact"] == cache_artifact
    assert result["cache_architecture_matrix"]["status"] == "pass"
    assert result["cache_architecture_matrix"]["missing_markers"] == [
        "test_mllm_stats_include_cache_fields"
    ]
    assert result["cache_architecture_matrix"]["missing_api_checks"] == [
        "turboquant_disk_roundtrip"
    ]
    assert result["cache_architecture_matrix"]["missing_api_command_markers"] == [
        "generic_turboquant_patcher_skips_hybrid_ssm"
    ]
    assert result["cache_architecture_matrix"]["missing_panel_markers"] == [
        "deepseek-v4 disables composite prefix cache by default even with stale cache config"
    ]
    assert result["cache_architecture_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS[-1]
    ]


def test_release_regression_manifest_requires_hybrid_ssm_companion_l2_contract():
    assert (
        "hybrid_ssm_companion_l2_contracts"
        in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS
    )


def test_release_regression_manifest_rejects_missing_current_cache_family_matrix(tmp_path):
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS
                        },
                        "missing_markers": [],
                        "missing_api_checks": [],
                        "missing_api_command_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["cache_architecture_matrix"]["missing_family_rows"] == list(
        EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
    )
    assert result["cache_architecture_matrix"]["failed_family_rows"] == list(
        EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
    )


def test_release_regression_manifest_rejects_incomplete_current_parser_registry_matrix(tmp_path):
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == parser_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": ["passes MiniMax through the registered minimax_m2 reasoning parser"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["parser_registry_matrix"]["artifact"] == parser_artifact
    assert result["parser_registry_matrix"]["status"] == "pass"
    assert result["parser_registry_matrix"]["missing_markers"] == [
        "passes MiniMax through the registered minimax_m2 reasoning parser"
    ]
    assert result["parser_registry_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_generation_defaults_matrix(tmp_path):
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == generation_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "test_request_output_caps_override_server_default_without_touching_context_cap"
                        ],
                        "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["generation_defaults_matrix"]["artifact"] == generation_artifact
    assert result["generation_defaults_matrix"]["status"] == "pass"
    assert result["generation_defaults_matrix"]["missing_markers"] == [
        "test_request_output_caps_override_server_default_without_touching_context_cap"
    ]
    assert result["generation_defaults_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_api_surface_matrix(tmp_path):
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_API_SURFACE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_nested_checks": ["request_output_caps_override_server_default"],
                        "missing_nested_markers": ["streaming_cache_detail_usage"],
                        "missing_panel_markers": [
                            "keeps Responses maxTokens as output budget only, never prompt context"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["api_surface_matrix"]["artifact"] == api_artifact
    assert result["api_surface_matrix"]["status"] == "pass"
    assert result["api_surface_matrix"]["missing_nested_checks"] == [
        "request_output_caps_override_server_default"
    ]
    assert result["api_surface_matrix"]["missing_nested_markers"] == [
        "streaming_cache_detail_usage"
    ]
    assert result["api_surface_matrix"]["missing_panel_markers"] == [
        "keeps Responses maxTokens as output budget only, never prompt context"
    ]
    assert result["api_surface_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_API_SURFACE_CHECKS[-1]
    ]


def test_release_regression_manifest_api_surface_requires_child_process_stdio_epipe_guard():
    assert "panel_child_process_stdio_epipe_guard" in EXPECTED_CURRENT_API_SURFACE_CHECKS


def test_release_regression_manifest_api_surface_requires_backend_stderr_split_epipe_guard():
    assert "panel_backend_stderr_split_epipe_guard" in EXPECTED_CURRENT_API_SURFACE_CHECKS


def test_release_regression_manifest_api_surface_requires_performance_health_epipe_guard():
    assert "panel_performance_health_epipe_guard" in EXPECTED_CURRENT_API_SURFACE_CHECKS


def test_release_regression_manifest_api_surface_requires_session_lifecycle_epipe_guard():
    assert "panel_session_lifecycle_epipe_guard" in EXPECTED_CURRENT_API_SURFACE_CHECKS


def test_release_regression_manifest_session_ipc_normalizes_epipe_lifecycle_errors():
    source = Path("panel/src/main/ipc/sessions.ts").read_text(encoding="utf-8")

    assert "function isExpectedSessionLifecycleDisconnectError" in source
    assert "function formatSessionLifecycleError" in source
    assert "Server connection lost. The model server may have stopped or restarted. Try restarting the session." in source
    assert "formatSessionLifecycleError(error)" in source
    assert "formatSessionLifecycleError(data.error)" in source


def test_release_regression_manifest_api_surface_requires_plain_attention_kv_status():
    assert "plain_attention_kv_status" in EXPECTED_CURRENT_API_SURFACE_CHECKS


def test_release_regression_manifest_api_surface_requires_named_cache_architecture_statuses():
    for check in (
        "dsv4_native_cache_status",
        "zaya_typed_cca_status",
        "hybrid_ssm_partial_reuse",
        "turboquant_kv_runtime_contract",
        "turboquant_disk_roundtrip",
        "no_generic_tq_on_hybrid_ssm",
    ):
        assert check in EXPECTED_CURRENT_API_SURFACE_CHECKS


def test_release_regression_manifest_rejects_incomplete_current_reasoning_template_matrix(tmp_path):
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == reasoning_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "test_dsv4_reasoning_effort_preserves_requested_rails"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["reasoning_template_matrix"]["artifact"] == reasoning_artifact
    assert result["reasoning_template_matrix"]["status"] == "pass"
    assert result["reasoning_template_matrix"]["missing_markers"] == [
        "test_dsv4_reasoning_effort_preserves_requested_rails"
    ]
    assert result["reasoning_template_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_tool_call_matrix(tmp_path):
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == tool_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_TOOL_CALL_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "panel max tool iterations caps tool loops"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["tool_call_matrix"]["artifact"] == tool_artifact
    assert result["tool_call_matrix"]["status"] == "pass"
    assert result["tool_call_matrix"]["missing_markers"] == [
        "panel max tool iterations caps tool loops"
    ]
    assert result["tool_call_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_TOOL_CALL_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_native_mtp_matrix(tmp_path):
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mtp_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_NATIVE_MTP_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "DSV4 additional args cannot reenable native MTP or deterministic sampling policy"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["native_mtp_matrix"]["artifact"] == mtp_artifact
    assert result["native_mtp_matrix"]["status"] == "pass"
    assert result["native_mtp_matrix"]["missing_markers"] == [
        "DSV4 additional args cannot reenable native MTP or deterministic sampling policy"
    ]
    assert result["native_mtp_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_NATIVE_MTP_CHECKS[-1]
    ]


def test_release_regression_manifest_rejects_incomplete_current_vl_media_matrix(tmp_path):
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == vl_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_VL_MEDIA_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_engine_markers": ["test_qwen36_jangtq_is_detected_as_vl"],
                        "missing_panel_markers": [
                            "keeps mxfp8 Qwen hybrid VLM multimodal"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["vl_media_matrix"]["artifact"] == vl_artifact
    assert result["vl_media_matrix"]["status"] == "pass"
    assert result["vl_media_matrix"]["missing_engine_markers"] == [
        "test_qwen36_jangtq_is_detected_as_vl"
    ]
    assert result["vl_media_matrix"]["missing_panel_markers"] == [
        "keeps mxfp8 Qwen hybrid VLM multimodal"
    ]
    assert result["vl_media_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_VL_MEDIA_CHECKS[-1]
    ]


def test_release_regression_manifest_vl_media_requires_engine_marker_completeness():
    assert "all_required_engine_markers_present" in EXPECTED_CURRENT_VL_MEDIA_CHECKS


def test_release_regression_manifest_rejects_incomplete_current_mcp_policy_matrix(tmp_path):
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS
                        },
                        "missing_engine_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mcp_artifact:
            checks = {name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS[:-1]}
            checks[EXPECTED_CURRENT_MCP_POLICY_CHECKS[-1]] = False
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": checks,
                        "missing_markers": [
                            "panel MCP import redacts managed metadata"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["mcp_policy_matrix"]["artifact"] == mcp_artifact
    assert result["mcp_policy_matrix"]["status"] == "pass"
    assert result["mcp_policy_matrix"]["missing_markers"] == [
        "panel MCP import redacts managed metadata"
    ]
    assert result["mcp_policy_matrix"]["failed_checks"] == [
        EXPECTED_CURRENT_MCP_POLICY_CHECKS[-1]
    ]


def test_release_regression_manifest_runner_embeds_current_proof_validation(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    api_cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "noheavy-api-cache-endpoint-runtime"
    ]
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    panel_tool_security_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "panel-tool-security-loop-boundary"
    ]
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    jang_model_compat_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
        "jang-model-compat-runtime-boundary"
    ]
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    release_surface_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = tmp_path / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == model_family_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "matched_rows": EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
                        "missing_rows": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == max_output_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS
                        },
                        "missing_markers": [],
                        "failed": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == packaged_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS
                        },
                        "failed": [],
                        "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == panel_tool_security_artifact:
            path.write_text(
                json.dumps({"status": "pass", "returncode": 0}) + "\n",
                encoding="utf-8",
            )
        elif artifact == jang_model_compat_artifact:
            path.write_text(
                json.dumps({"status": "pass", "failed": []}) + "\n",
                encoding="utf-8",
            )
        elif artifact == release_surface_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == model_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == cache_artifact:
            path.write_text(
                json.dumps(_passing_cache_architecture_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == parser_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == generation_artifact:
            path.write_text(
                json.dumps(_passing_generation_defaults_payload())
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_cache_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True
                            for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == api_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS
                        },
                        "missing_nested_checks": [],
                        "missing_nested_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == reasoning_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == tool_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mtp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == vl_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS
                        },
                        "missing_engine_markers": [],
                        "missing_panel_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif artifact == mcp_artifact:
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text('{"status":"pass","failed":[]}\n', encoding="utf-8")
    regression_suite = tmp_path / CURRENT_REGRESSION_SUITE_ARTIFACT
    regression_suite.parent.mkdir(parents=True, exist_ok=True)
    regression_suite.write_text(
        json.dumps(
            {
                "status": "pass",
                "failed_steps": [],
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
                "open_requirement_details": _passing_open_requirement_details(),
                **_passing_current_suite_progress_checkpoints(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_passing_covered_live_smoke_artifacts(tmp_path)
    _write_passing_covered_live_tool_smoke_artifacts(tmp_path)
    _write_expected_diagnostic_live_smoke_artifacts(tmp_path)
    _write_current_objective_digest(tmp_path)
    _write_passing_dev_ui_proof_artifacts(tmp_path)
    _write_passing_real_ui_live_model_proof_artifacts(tmp_path)
    _write_passing_real_ui_dsv4_memory_preflight_artifact(tmp_path)
    _write_passing_issue179_minimax_k_live_probe_memory_preflight(tmp_path)
    _write_passing_step37_vlm_runtime_audit(tmp_path)

    expected_real_ui_live_model_proof = validate_current_proof_sweep_artifacts(
        tmp_path
    )["real_ui_live_model_proof"]
    expected_real_ui_live_model_matrix = validate_current_proof_sweep_artifacts(
        tmp_path
    )["real_ui_live_model_matrix"]
    expected_issue175_179_release_boundary_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue175_179_release_boundary_audit"]
    expected_issue181_183_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue181_183_runtime_audit"]
    expected_public_app_issue_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["public_app_issue_audit"]
    expected_issue179_minimax_k_root_cause_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue179_minimax_k_root_cause_audit"]
    expected_issue179_minimax_k_live_probe_memory_preflight = (
        validate_current_proof_sweep_artifacts(tmp_path)[
            "issue179_minimax_k_live_probe_memory_preflight"
        ]
    )
    expected_installed_app_runtime_parity_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["installed_app_runtime_parity_audit"]
    expected_staged_app_runtime_parity_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["staged_app_runtime_parity_audit"]
    expected_issue175_177_installed_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue175_177_installed_runtime_audit"]
    expected_issue175_177_live_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["issue175_177_live_runtime_audit"]
    expected_real_ui_dsv4_memory_preflight = validate_current_proof_sweep_artifacts(
        tmp_path
    )["real_ui_dsv4_memory_preflight"]
    expected_dsv4_proof_artifact_freshness = validate_current_proof_sweep_artifacts(
        tmp_path
    )["dsv4_proof_artifact_freshness"]
    expected_step37_vlm_runtime_audit = validate_current_proof_sweep_artifacts(
        tmp_path
    )["step37_vlm_runtime_audit"]
    expected_mimo_v2_jang2l_root_cause = validate_current_proof_sweep_artifacts(
        tmp_path
    )["mimo_v2_jang2l_root_cause"]
    expected_objective_digest = validate_current_proof_sweep_artifacts(
        tmp_path
    )["objective_digest"]
    expected_release_blocker_ledger = validate_current_proof_sweep_artifacts(
        tmp_path
    )["release_blocker_ledger"]
    expected_component_ok = validate_current_proof_sweep_artifacts(
        tmp_path
    )["component_ok"]
    artifact = build_manifest_artifact(tmp_path)

    assert artifact["current_proof_sweep"] == {
        "status": "fail",
        "component_ok": expected_component_ok,
        "failed_components": [
            name for name, ok in expected_component_ok.items() if not ok
        ],
        "missing": [],
        "not_pass": [],
        "regression_suite": {
            "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
            "status": "pass",
            "failed_steps": [],
            "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "open_requirement_details": _passing_open_requirement_details(),
            "unexpected_open_requirements": [],
            "missing_expected_open_requirements": [],
            "open_requirement_detail_failures": [],
        },
        "objective_digest": expected_objective_digest,
        "mimo_v2_jang2l_root_cause": expected_mimo_v2_jang2l_root_cause,
        "model_family_matrix": {
            "artifact": model_family_artifact,
            "status": "pass",
            "matched_rows": list(EXPECTED_CURRENT_MODEL_FAMILY_ROWS),
            "missing_rows": [],
            "missing_expected_rows": [],
        },
        "model_artifact_matrix": {
            "artifact": model_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "cache_architecture_matrix": {
            "artifact": cache_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS},
            "missing_markers": [],
            "missing_api_checks": [],
            "missing_api_command_markers": [],
            "missing_panel_markers": [],
            "missing_family_rows": [],
            "failed_family_rows": [],
            "cache_family_matrix": _passing_cache_family_matrix(),
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "parser_registry_matrix": {
            "artifact": parser_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "generation_defaults_matrix": {
            "artifact": generation_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS},
            "missing_markers": [],
            "missing_family_rows": [],
            "failed_family_rows": [],
            "generation_defaults_family_matrix": _passing_generation_defaults_family_matrix(),
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "noheavy_api_cache_matrix": {
            "artifact": api_cache_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "api_surface_matrix": {
            "artifact": api_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_API_SURFACE_CHECKS},
            "missing_nested_checks": [],
            "missing_nested_markers": [],
            "missing_panel_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "reasoning_template_matrix": {
            "artifact": reasoning_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "tool_call_matrix": {
            "artifact": tool_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "panel_tool_security_matrix": {
            "artifact": panel_tool_security_artifact,
            "status": "pass",
            "returncode": 0,
            "missing": [],
        },
        "native_mtp_matrix": {
            "artifact": mtp_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "vl_media_matrix": {
            "artifact": vl_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS},
            "missing_engine_markers": [],
            "missing_panel_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "mcp_policy_matrix": {
            "artifact": mcp_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS},
            "missing_markers": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "max_output_context_matrix": {
            "artifact": max_output_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS},
            "missing_markers": [],
            "failed": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "jang_model_compat_matrix": {
            "artifact": jang_model_compat_artifact,
            "status": "pass",
            "failed": [],
            "missing": [],
        },
        "packaged_integrity_matrix": {
            "artifact": packaged_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS},
            "failed": [],
            "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "package_signing_preflight": {},
            "release_blockers": [],
            "unexpected_open_requirements": [],
            "missing_expected_open_requirements": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
        "release_surface_matrix": {
            "artifact": release_surface_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS},
            "failed_checks": [],
            "missing_expected_checks": [],
            "informational_false_checks": [],
        },
        "live_smoke_summaries": _expected_passing_covered_live_smoke_summaries(),
        "live_tool_smoke_summaries": (
            _expected_passing_covered_live_tool_smoke_summaries()
        ),
        "diagnostic_live_smoke_summaries": (
            _expected_passing_diagnostic_live_smoke_summaries()
        ),
        "dev_ui_proof": {
            "status": "pass",
            "artifacts": dict(CURRENT_DEV_UI_PROOF_ARTIFACTS),
            "missing": [],
            "failures": [],
        },
        "real_ui_live_model_proof": expected_real_ui_live_model_proof,
        "issue175_179_release_boundary_audit": expected_issue175_179_release_boundary_audit,
        "issue181_183_runtime_audit": expected_issue181_183_runtime_audit,
        "public_app_issue_audit": expected_public_app_issue_audit,
        "issue179_minimax_k_root_cause_audit": expected_issue179_minimax_k_root_cause_audit,
        "issue179_minimax_k_live_probe_memory_preflight": (
            expected_issue179_minimax_k_live_probe_memory_preflight
        ),
        "installed_app_runtime_parity_audit": expected_installed_app_runtime_parity_audit,
        "staged_app_runtime_parity_audit": expected_staged_app_runtime_parity_audit,
        "issue175_177_installed_runtime_audit": expected_issue175_177_installed_runtime_audit,
        "issue175_177_live_runtime_audit": expected_issue175_177_live_runtime_audit,
        "real_ui_live_model_matrix": expected_real_ui_live_model_matrix,
        "real_ui_dsv4_memory_preflight": expected_real_ui_dsv4_memory_preflight,
        "dsv4_proof_artifact_freshness": expected_dsv4_proof_artifact_freshness,
        "step37_vlm_runtime_audit": expected_step37_vlm_runtime_audit,
        "release_blocker_ledger": expected_release_blocker_ledger,
    }


def test_release_regression_manifest_runner_can_require_current_proof_sweep(tmp_path):
    artifact = build_manifest_artifact(tmp_path, require_current_proof_sweep=True)

    assert artifact["status"] == "fail"
    assert artifact["current_proof_sweep"]["missing"]


def test_release_regression_manifest_runner_fails_when_current_proof_sweep_fails_by_default(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "regression_suite": {
                "status": "open",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "mimo_v2_jang2l_runtime_quality_open",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["current_proof_sweep"]["status"] == "fail"
    assert artifact["release_ready"] is False


def test_release_regression_manifest_artifact_exposes_open_release_clearance(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "dsv4_long_output_code_exactness_open",
                        "status": "open",
                    },
                    {
                        "id": "real_ui_dsv4_memory_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path)

    assert artifact["status"] == "pass"
    assert artifact["release_ready"] is False
    assert artifact["current_proof_sweep"]["status"] == "pass"
    assert artifact["release_clearance"]["status"] == "open"
    assert artifact["release_clearance"]["proof_sweep_status"] == "pass"
    assert artifact["release_clearance"]["open_requirements"] == (
        EXPECTED_CURRENT_OPEN_REQUIREMENTS
    )
    blocker_ids = [
        blocker["id"] for blocker in artifact["release_clearance"]["blockers"]
    ]
    assert "dsv4_long_output_code_exactness_open" in blocker_ids
    assert "real_ui_dsv4_memory_blocked" in blocker_ids
    assert artifact["release_clearance"]["release_ready"] is False


def test_release_regression_manifest_release_ready_allows_only_deferred_sweep_failure(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner
    deferred_open_requirements = sorted(DEFERRED_RELEASE_OPEN_REQUIREMENTS)

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "failed_components": ["regression_suite"],
            "regression_suite": {
                "status": "open",
                "open_requirements": deferred_open_requirements,
            },
            "release_blocker_ledger": {
                "status": "pass",
                "blockers": [],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path, require_release_ready=True)

    assert artifact["status"] == "pass"
    assert artifact["release_ready"] is True
    assert artifact["release_clearance"]["status"] == "pass"
    assert artifact["release_clearance"]["proof_sweep_failure_is_deferred"] is True
    assert artifact["release_clearance"]["effective_open_requirements"] == []
    assert artifact["release_clearance"]["deferred_open_requirements"] == deferred_open_requirements


def test_release_regression_manifest_artifact_exposes_top_level_release_blockers(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "regression_suite": {
                "status": "open",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "mimo_v2_jang2l_runtime_quality_open",
                        "status": "open",
                    },
                    {
                        "id": "real_ui_dsv4_memory_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["release_ready"] is False
    assert [
        blocker["id"] for blocker in artifact["release_blockers"]
    ] == [
        "mimo_v2_jang2l_runtime_quality_open",
        "real_ui_dsv4_memory_blocked",
    ]
    assert artifact["release_blockers"] == artifact["release_clearance"]["blockers"]


def test_release_regression_manifest_runner_can_require_release_ready(monkeypatch, tmp_path):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "dsv4_long_output_code_exactness_open",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(tmp_path, require_release_ready=True)

    assert artifact["status"] == "fail"
    assert artifact["release_ready"] is False
    assert artifact["release_clearance"]["status"] == "open"


def test_release_regression_manifest_runner_prepackage_allows_only_packaging_blockers(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": [],
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "packaged_app_developer_id_signing_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(
        tmp_path,
        require_prepackage_ready=True,
    )

    assert artifact["status"] == "pass"
    assert artifact["prepackage_ready"] is True
    assert artifact["release_ready"] is False


def test_release_regression_manifest_runner_prepackage_allows_real_signing_only_sweep_failure(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "fail",
            "failed_components": ["no_release_blockers"],
            "component_ok": {
                "no_release_blockers": False,
                "regression_suite": True,
                "packaged_integrity_matrix": True,
            },
            "regression_suite": {
                "status": "pass",
                "open_requirements": [],
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "packaged_app_developer_id_signing_blocked",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(
        tmp_path,
        require_prepackage_ready=True,
    )

    assert artifact["status"] == "pass"
    assert artifact["prepackage_ready"] is True
    assert artifact["release_ready"] is False
    assert artifact["prepackage_clearance"]["proof_sweep_failed_components"] == [
        "no_release_blockers"
    ]


def test_release_regression_manifest_runner_prepackage_rejects_model_blockers(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_release_regression_manifest as runner

    monkeypatch.setattr(
        runner,
        "validate_current_proof_sweep_artifacts",
        lambda _root: {
            "status": "pass",
            "regression_suite": {
                "status": "pass",
                "open_requirements": [],
            },
            "release_blocker_ledger": {
                "status": "open",
                "blockers": [
                    {
                        "id": "dsv4_long_output_code_exactness_open",
                        "status": "open",
                    },
                ],
            },
        },
    )

    artifact = runner.build_manifest_artifact(
        tmp_path,
        require_prepackage_ready=True,
    )

    assert artifact["status"] == "fail"
    assert artifact["prepackage_ready"] is False
    assert artifact["release_ready"] is False


def test_release_regression_manifest_tracks_legacy_completions_output_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "Legacy /v1/completions max_tokens" in joined
    assert "per-request output cap" in joined
    assert "non-streaming and streaming" in joined
    assert "current-max-output-context-contract-20260522-chat-auto-server-default.json" in joined


def test_release_regression_manifest_tracks_generation_defaults_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["generation-defaults-no-hidden-forcing"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_generation_defaults_contract.py" in joined
    assert "current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json" in joined
    assert "current-generation-defaults-contract-20260602-step-greedy-display.json" not in joined
    assert "current-generation-defaults-contract-20260531-post-step-lfm-refresh.json" not in joined
    assert "current-generation-defaults-contract-20260526-settings-audit.json" not in joined
    assert "current-generation-defaults-contract-20260521.json" in joined
    assert "generation_config.json" in joined
    assert "jang_config.json" in joined
    assert "hidden sampler forcing" in joined
    assert "Additional Args cannot override app-owned reasoning, parser, cache, MTP" in joined
    assert "Step3p7 metadata-route" in joined


def test_release_regression_manifest_tracks_reasoning_template_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["reasoning-template-no-think-tag-leak"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_reasoning_template_contract.py" in joined
    assert "current-reasoning-template-contract-20260526-settings-audit.json" in joined
    assert "current-reasoning-template-contract-20260521.json" in joined
    assert "Reasoning on/off" in joined
    assert "think tags" in joined
    assert "Interleaved reasoning" in joined


def test_release_regression_manifest_tracks_api_surface_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["api-chat-responses-anthropic-ollama-parity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_api_surface_contract.py" in joined
    assert "current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json" in joined
    assert "current-api-surface-contract-20260602-cache-detail-zero-cached.json" not in joined
    assert "current-api-surface-contract-20260531-nested-epipe-childstream-refresh.json" not in joined
    assert "current-api-surface-contract-20260529-single-model-transition-lock.json" in joined
    assert "current-api-surface-contract-20260528-ollama-embeddings-single-model.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-nested-disconnect.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-request-end-guard.json" not in joined
    assert "current-api-surface-contract-20260528-epipe-socket-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-proxy-error-guard.json" not in joined
    assert "current-api-surface-contract-20260528-ollama-epipe-multiline-guard.json" not in joined
    assert "current-api-surface-contract-20260527-cache-endpoint-autoswitch-proof.json" not in joined
    assert "current-api-surface-contract-20260526-single-model-auto-switch-review.json" not in joined
    assert "current-api-surface-contract-20260525-single-model-responses-deltas.json" not in joined
    assert "current-api-surface-contract-20260525-single-model-ollama-chat-deltas.json" in joined
    assert "current-api-surface-contract-20260522-stream-cache-detail.json" in joined
    assert "destroyed client/backend socket" in joined
    assert "top-level request handler disconnect failures" in joined
    assert "EPIPE/ECONNRESET" in joined
    assert "Ollama-specific backend response streams guard response errors" in joined
    assert "OpenAI Chat Completions" in joined
    assert "OpenAI Responses" in joined
    assert "non-streaming and streaming" in joined
    assert "OpenAI legacy Completions" in joined
    assert "Anthropic" in joined
    assert "streaming max_tokens override" in joined
    assert "Ollama" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined
    assert "streaming num_predict" in joined
    assert "malformed num_predict" in joined
    assert "malformed context" in joined
    assert "cache_detail" in joined


def test_release_regression_manifest_tracks_current_defaults_reasoning_api_rechecks():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    generation = rows["generation-defaults-no-hidden-forcing"]
    generation_joined = " ".join(generation["commands"] + generation["artifacts"] + generation["proves"])
    assert "current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json" in generation_joined
    assert "current-generation-defaults-contract-20260602-step-greedy-display.json" not in generation_joined
    assert "current-generation-defaults-contract-20260531-post-step-lfm-refresh.json" not in generation_joined
    assert "current-generation-defaults-contract-20260526-settings-audit.json" not in generation_joined
    assert "current-generation-defaults-contract-20260522-recheck-no-hidden-forcing.json" in generation_joined
    assert "bundle max_new_tokens" in generation_joined
    assert "without hidden sampler or repetition floors" in generation_joined
    assert "server default output cap is not a request ceiling" in generation_joined
    assert "Step3p7 metadata-route" in generation_joined

    reasoning = rows["reasoning-template-no-think-tag-leak"]
    reasoning_joined = " ".join(reasoning["commands"] + reasoning["artifacts"] + reasoning["proves"])
    assert "current-reasoning-template-contract-20260526-settings-audit.json" in reasoning_joined
    assert "DSV4 requested reasoning-on/effort rails are preserved" in reasoning_joined
    assert "MiniMax and Ling family-specific reasoning boundaries" in reasoning_joined
    assert "tool follow-up reasoning state resets" in reasoning_joined

    api = rows["api-chat-responses-anthropic-ollama-parity"]
    api_joined = " ".join(api["commands"] + api["artifacts"] + api["proves"])
    assert "current-api-surface-contract-20260522-recheck-endpoint-assembly.json" in api_joined
    assert "server cache and DSV4 tool/parser surfaces stay named" in api_joined
    assert "panel request builders omit invalid persisted maxTokens" in api_joined


def test_release_regression_manifest_tracks_tool_calls_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["tool-call-loop-parser-cleanup"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_tool_call_contract.py" in joined
    assert "current-tool-call-contract-after-cross-model-loop-metrics-20260609.json" in joined
    assert "Tool parser residue" in joined
    assert "DSV4" in joined
    assert "maxToolIterations" in joined
    assert "live DSV4 write_file DSML degradation is repaired schema-safely" in joined
    assert "current-tool-call-contract-20260522-dsv4-live-write-file-repair.json" in joined
    assert "current-dsv4-default-cache-tool-loop-thinking-on-20260525.json" in joined
    assert "explicit-thinking DSV4 default-cache tool-loop diagnostic remains review" in joined
    assert "current-dsv4-default-cache-tool-loop-poolon-materialized-parserfix-20260522/result.json" in joined
    assert "default-cache live loop reaches three tools but still fails exact WebGLRenderer fidelity" in joined


def test_release_regression_manifest_tracks_panel_cache_family_gating():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["panel-session-cache-settings-family-gating"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "cache_architecture"
    assert "run_noheavy_panel_settings_contract.py" in joined
    assert "current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json" in joined
    assert "current-panel-settings-contract-proof-20260524-text-additional-args-sanitizer.json" in joined
    assert "current-panel-settings-contract-proof-20260522-launch-memory-warning.json" in joined
    assert "DSV4 pool quant" in joined
    assert "JANG/JANGTQ/MXFP" in joined
    assert "stale Additional Args cannot override app-owned server, template, model-name, MCP, max output/context" in joined
    assert "warning-only for lazy-mmap" in joined


def test_release_regression_manifest_tracks_mcp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["mcp-policy-ui-gateway"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_mcp_policy_contract.py" in joined
    assert "current-mcp-policy-contract-20260531-post-step-lfm-refresh.json" in joined
    assert "MCP autodiscovery" in joined
    assert "redaction" in joined
    assert "gateway routing" in joined
    assert "panel config source" in joined
    assert "required marker" in joined


def test_release_regression_manifest_tracks_current_mcp_and_vl_media_rechecks():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    mcp = rows["mcp-policy-ui-gateway"]
    mcp_joined = " ".join(mcp["commands"] + mcp["artifacts"] + mcp["proves"])
    assert "current-mcp-policy-contract-20260522-recheck-ui-gateway.json" in mcp_joined
    assert "mcp.json/jsonc import" in mcp_joined
    assert "built-in Electron tools remain separate from MCP execution" in mcp_joined
    assert "ambiguous multi-session MCP requests" in mcp_joined

    vl_media = rows["vl-media-cache-tool-followup"]
    vl_joined = " ".join(vl_media["commands"] + vl_media["artifacts"] + vl_media["proves"])
    assert "current-vl-media-cache-contract-20260522-recheck-qwen-zaya-media.json" in vl_joined
    assert "Qwen3.6 VL JANG indexed-MTP" in vl_joined
    assert "MXTQ/JANGTQ and MXFP4/MXFP8 Qwen VLM" in vl_joined
    assert "hybrid SSM companion cache" in vl_joined
    assert "read_video" in vl_joined
    assert "image/video tool-result follow-up" in vl_joined


def test_release_regression_manifest_tracks_qwen_jang_live_speed_review():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["qwen-jang-mx-live-speed-review"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "performance"
    assert row["mode"] == "live"
    assert "current-decode-speed-live-qwen27-jang4m-20260522-hybrid-tq-review.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-source-20260606.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-packaged-keepalloc-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-installed-app-deterministic-pp-20260606.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-source-bypass-fix-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-installed-app-deterministic-pp-20260606.json" in joined
    assert "MLX and MLX-metal wheel tags" in joined
    assert "selective live TurboQuant for Qwen attention KV layers" in joined
    assert "Post norm-format loader fix live Qwen MTP/VLM default-cache row clears coherency" in joined
    assert "SingleBatchGenerator honors the explicit prefill keep-alloc CLI/env path" in joined
    assert "Compat bundled MTP prefill is retained as the user-facing packaged regression diagnostic" in joined
    assert "native-MTP MLLM/VL prefill remains below the floor" in joined
    assert "prompt-processing floor compares isolated native-MTP/VL prefill against the isolated text-loader baseline" in joined


def test_release_regression_manifest_tracks_packaged_integrity_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["packaged-release-integrity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_packaged_integrity_contract.py" in joined
    assert "current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json" in joined
    assert "current-packaged-integrity-contract-20260601-qwen-fix-resigned-staged-app.json" not in joined
    assert "current-packaged-integrity-contract-20260601-cache-ipc-epipe-package-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260531-after-lfm2-staged-sync.json" not in joined
    assert "current-packaged-integrity-contract-20260531-step37-mixed-swa-runtime.json" not in joined
    assert (
        "build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json"
        in " ".join(row["commands"])
    )
    assert "current-packaged-integrity-contract-20260531-local-release-decision-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260531-gemma4-l2-rotating-kv-fix.json" not in joined
    assert "current-packaged-integrity-contract-20260531-childstream-epipe-refresh.json" not in joined
    assert "current-packaged-integrity-contract-20260530-bundled-sync-after-step37-projector.json" not in joined
    assert "current-packaged-integrity-contract-20260521.json" not in joined
    assert "Version triples" in joined
    assert "bundled Python hash parity" in joined
    assert "objective proof digest" in joined
    assert (
        "current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json"
        in joined
    )
    assert "objective-gate-enforced" in joined
    assert "verify-bundled" in joined


def test_release_regression_manifest_tracks_current_packaged_integrity_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["packaged-release-integrity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json" in joined
    assert "current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json" in joined
    assert "current-packaged-integrity-contract-20260601-qwen-fix-resigned-staged-app.json" not in joined
    assert "current-packaged-integrity-contract-20260524-text-additional-args-sanitizer.json" in joined
    assert "clean JANG source path" in joined
    assert "bundled critical jang_tools files match source content" in joined
    assert "console-script shebangs are relocatable" in joined
    assert "fails only on the known objective rows" in joined


def test_release_regression_manifest_tracks_public_release_surface_preflight():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["public-release-surface-preflight"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "release_surface"
    assert "run_release_surface_contract.py" in joined
    assert (
        CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
        == "build/current-release-surface-contract-20260602-v154-live-public-after-site-fix.json"
    )
    assert "current-release-surface-contract-20260602-v154-live-public-after-site-fix.json" in joined
    assert "latest.json" in joined
    assert "PyPI" in joined
    assert "GitHub release" in joined
    assert "updater" in joined.lower()
    assert "post-release-updater" in joined
    assert "published updater state is complete" in joined
    assert "--live-public" in joined
    assert "mlx.studio/update/latest.json" in joined
    assert "PyPI files" in joined
    assert "no-store/no-cache headers" in joined
    assert "source repo release tag" in joined
    assert "current-release-surface-contract-20260524-live-source-tag-parity.json" in joined


def test_release_regression_manifest_tracks_current_updater_and_i18n_rechecks():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    panel = rows["panel-session-cache-settings-family-gating"]
    panel_joined = " ".join(panel["commands"] + panel["artifacts"] + panel["proves"])
    assert "current-panel-settings-contract-proof-20260522-recheck-update-notice-i18n.json" in panel_joined
    assert "What's New update notice" in panel_joined
    assert "MCP, MTP, latest.json, Developer ID, L2 disk cache, and max_tokens" in panel_joined
    assert "all five locales" in panel_joined

    release = rows["public-release-surface-preflight"]
    release_joined = " ".join(release["commands"] + release["artifacts"] + release["proves"])
    assert "current-release-surface-contract-20260522-recheck-updater-i18n.json" in release_joined
    assert "source version consistent across pyproject.toml and panel/package.json" in release_joined
    assert "complete post-release updater state" in release_joined
    assert "incomplete bumped latest.json" in release_joined

    ling = rows["ling-bailing-multilingual-quality-live"]
    ling_joined = " ".join(ling["commands"] + ling["artifacts"] + ling["proves"])
    assert (
        "current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json"
        in ling_joined
    )


def test_release_regression_manifest_tracks_live_only_boundaries():
    manifest = build_manifest()
    live_rows = [row for row in manifest["rows"] if row["mode"] == "live"]
    live_ids = {row["id"] for row in live_rows}

    assert "dsv4-long-output-quality-live" in live_ids
    assert "model-family-live-multiturn-soak" in live_ids
    assert all(row["heavy"] for row in live_rows)


def test_release_regression_manifest_tracks_fresh_dsv4_live_failure_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["dsv4-long-output-quality-live"]
    joined = " ".join(row["artifacts"] + row["proves"])

    assert "current-dsv4-route-mode-code-exactness-chat-off-user-ram-override-20260606.json" in joined
    assert "current-dsv4-route-mode-code-exactness-chat-on-user-ram-override-20260606.json" in joined
    assert "current-dsv4-route-mode-code-exactness-bundled-after-bundle-refresh-20260606.json" in joined
    assert "current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json" in joined
    assert "current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json" in joined
    assert "current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json" in joined
    assert "current-dsv4-route-mode-code-exactness-existing-server-chatmax-20260524.json" in joined
    assert "current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json" in joined
    assert "current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json" in joined
    assert "current-dsv4-prompt-boundary-bisection-live-20260524-1317.json" in joined
    assert "current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json" in joined
    assert "current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json" in joined
    assert "current-dsv4-scene-token-rank-contrast-live-20260524-1324.json" in joined
    assert "current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json" in joined
    assert "current-dsv4-hidden-reasoning-control-live-20260524-1335.json" in joined
    assert "current-dsv4-template-parity-diagnostic-20260524-1343.json" in joined
    assert "current-dsv4-jang-prefill-execution-variant-logits-20260524.json" in joined
    assert "current-dsv4-jang-prompt-variant-logit-probe-20260524.json" in joined
    assert "current-dsv4-reasoning-policy-live-20260524-1408.json" in joined
    assert "current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json" in joined
    assert "current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json" in joined
    assert "current-dsv4-jang-thinking-off-logit-probe-20260524.json" in joined
    assert "current-dsv4-jang-live-api-copy-framing-canary-20260524.json" in joined
    assert "current-dsv4-route-mode-code-exactness-preflight-after-mimo-classifier-refresh-20260608.json" in joined
    assert "current-dsv4-route-mode-code-exactness-memory-preflight-20260603-second-local-check.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-memory-preflight-20260602-developer-id-local-recheck.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-memory-preflight-20260601-post-epipe-fix.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-memory-preflight-20260601-local-recheck.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-memory-preflight-20260601-post-wrapped-epipe.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-memory-preflight-20260531-release-decision-refresh.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-live-memory-preflight-20260531.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-post-install-sync.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-continue-refresh.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0625.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260528-0600.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260526.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-source-memory-preflight-20260525.json" not in joined
    assert "current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json" in joined
    assert "current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json" in joined
    assert "current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json" in joined
    assert "current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json" in joined
    assert "current-dsv4-route-mode-code-exactness-direct-off-no-punct-dryrun-20260525.json" in joined
    assert "current-dsv4-route-mode-code-exactness-direct-off-no-punct-preflight-20260525.json" in joined
    assert "identifier integrity" in joined
    assert "explicit thinking-off still fails exact code" in joined
    assert "repetition_penalty=1.0 live route controls still fail direct/off" in joined
    assert "true-bundled JANGTQ-K direct/off recheck still fails" in joined
    assert "chat_max remains open" in joined
    assert "budget, stops, requested thinking, Responses, and completion route" in joined
    assert "explicit DSV4 thinking-off produces visible output" in joined
    assert "prompt-boundary bisection" in joined
    assert "colon-vs-period logprob trace" in joined
    assert "scene token rank contrast" in joined
    assert "hidden reasoning" in joined
    assert "template parity" in joined
    assert "prefill execution variants" in joined
    assert "prompt wording changes identifier logits" in joined
    assert "batch generator logit divergence" in joined
    assert "source full-output exactness is still missing because local memory preflight skips" in joined
    assert "no-punctuation direct/off variant is queued under thinking_closed" in joined


def test_release_regression_manifest_tracks_ling_multilingual_quality_clearance_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["ling-bailing-multilingual-quality-live"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])

    assert row["domain"] == "reasoning_template"
    assert row["mode"] == "live"
    assert row["heavy"] is True
    assert "Ling/Bailing multilingual output quality is release-cleared" in joined
    assert "current-production-family-live-ling-bundled-current-20260606.json" in joined
    assert "run_production_family_audit.py --rows ling_flash_tq --live --py /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12" in joined
    assert "current-ling-jangtq-strict-russian-nocache-bundled-4850c9c2-20260524.json" in joined
    assert "current-ling-mxfp4-crack-strict-russian-nocache-bundled-4850c9c2-20260524.json" in joined
    assert "current-ling-jangtq-russian-prompt-variant-probe-20260524.json" in joined
    assert "current-ling-jangtq-server-repeat-russian-source-prefill-stream-20260524.json" in joined
    assert "current-ling-jangtq-server-repeat-russian-bundled-prefill-stream-20260524.json" in joined
    assert "current-ling-jangtq-server-repeat-russian-bundled-native-prefill-stream-20260524.json" in joined
    assert "current-production-family-live-ling-bundled-native-rerun-20260524.json" in joined
    assert "current-ling-jangtq-cold-skipcache-repeat-bundled-native-20260524.json" in joined
    assert "current-ling-jangtq-batchgen-temp0-repeat-bundled-native-20260524.json" in joined
    assert "current-ling-jangtq-simple-engine-control-bundled-after-mpp-fix-20260524.json" in joined
    assert "current-ling-jangtq-continuous-control-bundled-after-mpp-fix-20260524.json" in joined
    assert "current-production-family-live-ling-bundled-after-mpp-fix-20260524.json" in joined
    assert "current-production-family-live-ling-bundled-after-topk-policy-20260524.json" in joined
    assert "clearance artifacts are zero-CJK" in joined
    assert "Older failing artifacts remain tracked as regression evidence" in joined
    assert "current open row" not in joined
    assert "CJK leakage remains release-visible" not in joined


def test_release_regression_manifest_tracks_gemma4_crack_language_visible_quality_boundary():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["gemma4-crack-live-language-visible-quality"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])

    assert row["domain"] == "reasoning_template"
    assert row["mode"] == "live"
    assert row["heavy"] is True
    assert "Gemma-4-26B-A4B JANG_4M CRACK" in joined
    assert "switch_mlp experts" in joined
    assert "dense 31B JANG_4M MTP comparison row" in joined
    assert "Bundled Python must import jang_tools" in joined
    assert "visible English output with preserved identifiers" in joined
    assert "max_thinking_tokens=16 artifact remains diagnostic" in joined
    assert "CJK/Korean/Chinese leakage" in joined
    assert "affine/JANG matmul dispatch" in joined
    assert "Gemma4 switch_mlp remapping" in joined
    assert "mixed_swa_kv_v1" in joined
    assert "80 tok/s decode floor" in joined
    assert "generic live TurboQuant KV" in joined
    assert "generic TurboQuant KV off" in joined
    assert "source now proves native mixed-SWA cache telemetry as paged+mixed_swa" in joined
    assert "clears the sustained 512-token speed-floor prompt" in joined
    assert "Current installed-app speed clearance requires source-hash parity, paged+mixed_swa cache telemetry" in joined
    assert "stream UI TPS above 80 tok/s for cold and cache-hit rows" in joined
    assert "cold-wall TTFT tracking" in joined
    assert "decode_tok_s_stream" in joined
    assert "does not supersede older sub-floor rows until the current issue115 speed-floor artifact exists and passes" in joined
    assert "Current installed app proves source-hash parity" not in joined
    assert "installed app speed remains a release risk" not in joined
    assert "wall decode samples are below 80 tok/s" not in joined
    assert "installed repeat still remains below or unstable" not in joined
    assert "compat MLX wheels remain a separate Sequoia-flavor speed/quality risk" in joined
    assert "staged Sequoia app uses macosx_14_0_arm64 MLX/Metal wheels" in joined
    assert "staged Tahoe app uses macosx_26_0_arm64 MLX/Metal wheels" in joined
    assert "/Applications/vMLX.app runtime flavor is checked separately" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-contract-20260524.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-source-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json" in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-native-schema-v5-20260525.json" not in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-installed-app-skip-redundant-store-20260525.json" not in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-speedram-945c84ba-20260524b.json" not in joined
    assert "current-runtime-memory-stress-gemma4-26b-jang4m-text-installed149.json" not in joined
    assert "current-production-family-live-gemma4-crack-parser-tool-config-latest-jang-20260524.json" in joined


def test_release_regression_manifest_gemma4_installed_wheel_claim_matches_installed_app():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["gemma4-crack-live-language-visible-quality"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])
    site_packages = Path(
        "/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/python3.12/site-packages"
    )
    mlx_wheel = (site_packages / "mlx-0.31.2.dist-info/WHEEL").read_text(
        encoding="utf-8"
    )
    mlx_metal_wheel = (
        site_packages / "mlx_metal-0.31.2.dist-info/WHEEL"
    ).read_text(encoding="utf-8")

    if "Tag: cp312-cp312-macosx_26_0_arm64" in mlx_wheel:
        assert "Tag: py3-none-macosx_26_0_arm64" in mlx_metal_wheel
        assert "installed Tahoe-native app now uses macosx_26_0_arm64" in joined
    else:
        assert "Tag: cp312-cp312-macosx_14_0_arm64" in mlx_wheel
        assert "Tag: py3-none-macosx_14_0_arm64" in mlx_metal_wheel
        assert "installed Sequoia-compatible app now uses macosx_14_0_arm64" in joined
    assert "installed app currently uses macosx_14_0_arm64" not in joined


def test_release_regression_manifest_clears_gemma4_ui_speed_without_hiding_cold_wall_ttft():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    row = rows["gemma4-crack-live-language-visible-quality"]
    joined = " ".join(row["proves"] + row["commands"] + row["artifacts"])
    installed_speed_artifacts = [
        artifact
        for artifact in row["artifacts"]
        if "installed-app" in artifact and "speed-floor" in artifact
    ]

    current_artifact = Path(
        "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json"
    )
    if not current_artifact.exists():
        assert str(current_artifact) in joined
        assert "Current installed-app speed clearance requires" in joined
        assert "does not supersede older sub-floor rows until the current issue115 speed-floor artifact exists and passes" in joined
        assert "Current installed app proves source-hash parity" not in joined
        return

    payload = json.loads(current_artifact.read_text())
    stream_tps = [
        result["stream_speed"]["decode_tok_s_stream"]
        for result in payload["results"]
    ]
    cold_wall = payload["results"][0]["speed"]["decode_tok_s_wall"]

    assert min(stream_tps) >= 80.0
    assert cold_wall < 80.0
    assert "stream UI TPS above 80 tok/s for cold and cache-hit rows" in joined
    assert "cold wall decode includes TTFT and remains tracked separately" in joined
    assert "installed app speed remains a release risk" not in joined


def test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    command_text = " ".join(row["commands"]).lower()
    proves_text = " ".join(row["proves"]).lower()

    if "qwen mtp" in proves_text and "no-heavy" not in proves_text:
        assert "qwen" in command_text and "mtp" in command_text


def test_release_regression_manifest_live_soak_targets_present_qwen_hybrid_row():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    command_text = " ".join(row["commands"])

    assert "qwen36_moe_crack" in command_text
    assert "qwen36_moe_tq4" not in command_text


def test_release_regression_manifest_tracks_family_parser_cli_choice_guard():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "CLI-accepted parser choice" in joined
    assert "current-model-family-detection-contract-20260522-plain-kv-cache-health.json" in joined


def test_release_regression_manifest_tracks_decode_speed_artifact_format_matrix():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "DSV4 native composite" in joined
    assert "generic JANGTQ/MXTQ" in joined
    assert "current-model-family-detection-contract-20260522-plain-kv-cache-health.json" in joined
    assert "row parser/modality policy" in joined
    assert "forced JANGTQ acceleration" in joined
    assert "JANG-only MX matmul rows stay text-only launch rows" in joined
    assert "current-model-family-detection-contract-20260522-jang-only-mx-matmul-policy.json" in joined
    assert "Mistral JANGTQ" in joined
    assert "Mistral MXFP4" in joined
    assert "GPT-OSS" in joined
    assert "Nemotron 3 JANGTQ2" in joined


def test_release_regression_manifest_tracks_qwen_nemotron_hybrid_cache_rows():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "Qwen dense linear-attention wrappers stay hybrid cache" in joined
    assert "Qwen MoE text linear-attention wrappers stay hybrid cache" in joined
    assert "base Nemotron-H registry rows stay hybrid cache" in joined
    assert "current-model-family-detection-contract-20260522-qwen-nemotron-hybrid-cache.json" in joined
    assert "Nemotron parser/reasoning" in joined
    assert "MXFP4" in joined
    assert "plain KV JANG/JANGTQ/MXFP rows" in joined
    assert "local high-risk DSV4" in joined
    assert "Nemotron Omni/Nano" in joined
    assert "current-model-family-detection-contract-20260522-local-artifact-registry.json" in joined
    assert "Panel detection matches the same current local high-risk paths" in joined
    assert "native-MTP Qwen affine-JANG VL routing" in joined
    assert "current-model-family-detection-contract-20260522-panel-local-paths.json" in joined
    assert "Qwen 3.6 release rows intentionally keep qwen3_5/qwen3.5 family aliases" in joined
    assert "current-model-family-detection-contract-20260522-qwen36-alias.json" in joined


def test_release_regression_manifest_tracks_zaya_stale_stamp_policy():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "stale ZAYA converter stamps" in joined
    assert "cannot disable reasoning" in joined
    assert "current-model-family-detection-contract-20260522-zaya-stale-stamp.json" in joined


def test_release_regression_manifest_tracks_panel_family_launch_wiring():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-model-family-detection-contract-20260522-panel-launch-wiring.json" in joined
    assert "current-model-family-detection-contract-20260522-zaya-hy3-qwen-vl-profile-rows.json" in joined
    assert "Panel session launch builder" in joined
    assert "MiniMax minimax_m2" in joined
    assert "Qwen3.6 hybrid cache forces paged cache" in joined
    assert "ZAYA qwen3 reasoning parser" in joined
    assert "DSV4 stale cache/additionalArgs suppression" in joined
    assert "native-MTP D3 launch policy" in joined
    assert "ZAYA1-VL JANGTQ_K/JANGTQ2/JANGTQ4 plain-template no-reasoning capability truth" in joined
    assert "Hy3 JANGTQ_K Low/High reasoning contract" in joined
    assert "affine-JANG Qwen native-MTP VL/video" in joined


def test_release_regression_manifest_tracks_mxfp_vlm_loader_quant_mode():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "MXFP4/MXFP8 VLM loader" in joined
    assert "declared quantization mode" in joined
    assert "family-specific loader wrappers" in joined
    assert "current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json" in joined
    assert "current-model-artifact-format-contract-20260522-loader-wrapper-hashes.json" in joined


def test_release_regression_manifest_tracks_affine_jang_loader_acceptance():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "generic affine JANG loader" in joined
    assert "weight_format=affine" in joined
    assert "current-model-artifact-format-contract-20260522-affine-jang-loader.json" in joined


def test_release_regression_manifest_commands_are_declared_for_noheavy_rows():
    manifest = build_manifest()
    noheavy_rows = [row for row in manifest["rows"] if row["mode"] == "noheavy"]

    assert noheavy_rows
    for row in noheavy_rows:
        assert row["commands"], row["id"]


def test_release_regression_manifest_runner_commands_exist():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for command in row["commands"]:
            if "tests/cross_matrix/" not in command:
                continue
            runner = command.split("tests/cross_matrix/", 1)[1].split()[0]
            path = Path("tests/cross_matrix") / runner
            assert path.exists(), f"{row['id']} references missing runner {path}"


def test_release_regression_manifest_python_entrypoints_are_tracked():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for command in row["commands"]:
            for token in shlex.split(command):
                if not token.endswith(".py"):
                    continue
                path = Path(token)
                if path.is_absolute():
                    continue
                assert path.exists(), f"{row['id']} references missing Python entrypoint {path}"


def test_release_regression_manifest_python_runner_commands_are_executable_invocations():
    manifest = build_manifest()

    for row in manifest["rows"]:
        for command in row["commands"]:
            parts = shlex.split(command)
            for index, token in enumerate(parts):
                if not token.startswith("tests/cross_matrix/") or not token.endswith(".py"):
                    continue
                launcher = parts[:index]
                assert launcher, f"{row['id']} references {token} without a Python launcher"
                assert (
                    "python" in launcher
                    or "python3" in launcher
                    or any(part.endswith("/python") or part.endswith("/python3") for part in launcher)
                ), f"{row['id']} references {token} without an executable Python command"


def test_release_regression_manifest_tracks_model_artifact_detection_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_model_artifact_format_contract.py" in joined
    assert "current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json" in joined
    assert "JANGTQ" in joined
    assert "MXFP4" in joined
    assert "MXFP8" in joined
    assert "MTP" in joined


def test_release_regression_manifest_tracks_current_jang_mxfp_mtp_loader_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-artifact-format-detection"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-model-artifact-format-contract-20260522-recheck-jang-mxfp-mtp-loaders.json" in joined
    assert "JANGTQ_K mixed routed bits use the down-projection bit width for gather_dn" in joined
    assert "Ling/Bailing flat 2D switch_mlp repair and correct 3D no-op stay covered" in joined
    assert "Qwen plain MLX 4bit stays distinct from JANG, JANGTQ/MXTQ, MXFP4, and MXFP8" in joined
    assert "native-MTP detection uses real indexed weights rather than path names" in joined


def test_release_regression_manifest_tracks_named_model_family_detection_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "model_family_detection"
    assert "run_model_family_detection_contract.py" in joined
    assert "current-model-family-detection-contract-after-n2-policy-row-20260609.json" in joined
    assert "current-model-family-detection-contract-20260602-step-jangtq-boundary.json" not in joined
    assert "current-model-family-detection-contract-20260531-post-step-lfm-refresh.json" not in joined
    assert "current-model-family-detection-contract-20260522-plain-kv-cache-health.json" in joined
    assert "DSV4" in joined
    assert "ZAYA" in joined
    assert "Ling" in joined
    assert "Nemotron" in joined
    assert "Qwen 3.6" in joined
    assert "MXFP4" in joined
    assert "MXFP8" in joined
    assert "JANG-only" in joined
    assert "Mistral JANGTQ" in joined
    assert "Mistral MXFP4" in joined
    assert "Nemotron 3 JANGTQ2" in joined
    assert "GPT-OSS" in joined
    assert "JANGTQ/MXTQ" in joined
    assert "registered engine parser" in joined
    assert "CLI-accepted parser choice" in joined
    assert "MiniMax" in joined
    assert "Hy3" in joined


def test_release_regression_manifest_tracks_current_family_launch_policy_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-detection-noheavy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-model-family-detection-contract-20260522-recheck-launch-policy-matrix.json" in joined
    assert "engine, panel, and session launch wiring all run in the same family launch-policy matrix" in joined
    assert "ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron-H, Qwen 3.6, MiniMax, Hy3, and DSV4 rows keep aligned parser/cache/modality policy" in joined
    assert "session launch strips stale DSV4 additionalArgs and preserves native-MTP D3 only where supported" in joined


def test_release_regression_manifest_tracks_parser_parity_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["parser-registry-tool-reasoning-parity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_parser_registry_contract.py" in joined
    assert "current-parser-registry-contract-20260522-reasoning-dropdown.json" in joined
    assert "current-parser-registry-contract-20260522-non-reasoning-boundaries.json" in joined
    assert "MiniMax" in joined
    assert "reasoning parser" in joined
    assert "Reasoning parser dropdown covers every parser" in joined
    assert "CLI accepts every registered reasoning parser" in joined
    assert "tool parser" in joined
    assert "Qwen2/Qwen2-VL, Gemma 3, and GLM base stay off reasoning rails" in joined


def test_release_regression_manifest_tracks_max_output_context_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_max_output_context_contract.py" in joined
    assert "current-max-output-context-contract-20260522-chat-auto-server-default.json" in joined
    assert "Server Default Max Output Tokens" in joined
    assert "Chat Max Output Tokens" in joined
    assert "non-streaming and streaming" in joined
    assert "Legacy /v1/completions max_tokens" in joined
    assert "non-streaming and streaming" in joined
    assert "Prompt/context aliases clamp" in joined
    assert "Ollama gateway non-stream num_predict remains request-scoped" in joined
    assert "current-max-output-context-contract-20260523-ollama-nonstream-caps.json" in joined
    assert "malformed num_predict" in joined
    assert "malformed context" in joined
    assert "Max Context Tokens" in joined
    assert "--max-tokens" in joined
    assert "--max-prompt-tokens" in joined
    assert "coding-tool configs" in joined
    assert "Auto chat Max Tokens omits per-request output caps" in joined


def test_release_regression_manifest_tracks_current_server_chat_output_compat_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["chat-settings-max-output-context-ui"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-max-output-context-contract-20260522-recheck-server-chat-output-compat.json" in joined
    assert "explicit per-request Chat/Responses output caps can be below or above Server Default Max Output Tokens" in joined
    assert "later Auto Chat/Responses requests still use the original server startup default" in joined
    assert "server startup maxTokens explicitness survives wake/reload and CLI implicit startup paths" in joined


def test_release_regression_manifest_tracks_vl_media_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["vl-media-cache-tool-followup"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_vl_media_cache_contract.py" in joined
    assert "current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json" in joined
    assert "current-vl-media-cache-contract-20260601-qwen3vl-frame-list-fallback.json" not in joined
    assert "current-vl-media-cache-contract-20260522-panel-family.json" in joined
    assert "VLM media request serialization" in joined
    assert "media cache salting" in joined
    assert "tool follow-up" in joined
    assert "Qwen VL/video/hybrid" in joined
    assert "affine-JANG Qwen" in joined
    assert "Nemotron stale-Omni" in joined
    assert "video" in joined.lower()


def test_release_regression_manifest_tracks_cache_architecture_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["cache-architecture-family-classification"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_cache_architecture_contract.py" in joined
    assert "current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json" in joined
    assert "current-cache-architecture-contract-20260602-step-jangtq-boundary.json" not in joined
    assert "current-cache-architecture-contract-20260601-zaya-dsv4-terminal-disk-guard.json" not in joined
    assert "current-cache-architecture-contract-20260601-step37-mixed-swa-ui-storage-quant.json" not in joined
    assert "current-cache-architecture-contract-20260530-lfm2-tool-parser-local.json" in joined
    assert "current-cache-architecture-contract-20260528-gemma4-mixed-swa-row.json" in joined
    assert "current-cache-architecture-contract-20260527-cache-family-matrix.json" in joined
    assert "current-cache-architecture-contract-20260524-openai-single-model-streaming-audit.json" in joined
    assert "current-cache-architecture-contract-20260521.json" in joined
    assert "DSV4 composite" in joined
    assert "DSV4 SWA/HCA/CSA native composite cache components" in joined
    assert "Ling/Bailing" in joined
    assert "Qwen3.6" in joined
    assert "ZAYA CCA" in joined
    assert "hybrid SSM" in joined
    assert "TurboQuant KV" in joined
    assert "MLA" in joined
    assert "current-cache-architecture-contract-20260522-panel-cache-launch.json" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-quant-append.json" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-timing.json" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-env-gate.json" in joined
    assert "DSV4 pool quant reads reuse a materialized pool view" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-materialized-cache.json" in joined
    assert "pool quant codec appends only newly generated CSA/HCA pool rows" in joined
    assert "DSV4 panel env mapping enables pool quant by default only under DSV4 native composite cache" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json" in joined
    assert "DSV4 timing probe covers prefix-cache replay and cold-store boundaries" in joined
    assert "Panel session launch builder preserves DSV4 default-on native prefix-cache policy" in joined
    assert "Qwen3.6 hybrid and Mamba paged-cache forcing" in joined
    assert "regular KV stale saved false semantics" in joined
    assert "Gemma4" in joined
    assert "mixed-SWA" in joined


def test_release_regression_manifest_tracks_native_mtp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["native-mtp-d3-effect-policy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_native_mtp_contract.py" in joined
    assert "current-native-mtp-contract-after-noheavy-contract-refresh-20260608.json" in joined
    assert "current-qwen36-mtp-gdn-sink-source-proof-20260606.json" in joined
    assert "current-qwen35-dense-mtp-gdn-sink-fix-20260606.json" in joined
    assert "gdn_sink" in joined
    assert "current-native-mtp-contract-20260522-config-only.json" in joined
    assert "current-native-mtp-contract-20260522-dsv4-additional-args.json" in joined
    assert "Native MTP D3" in joined
    assert "DSV4" in joined
    assert "stale additionalArgs" in joined
    assert "deterministic MTP sampling" in joined
    assert "Config-only MTP bundles" in joined
    assert "indexed mtp.* tensors" in joined
    assert "decode speed and output equivalence" in joined
    assert "prompt-processing floor" in joined
    assert "current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606/result.json" in joined
    assert "same-artifact AR-vs-MTP" in joined
    assert "MTP decode is not the MLLM prefill bottleneck" in joined
    assert "prefill" in joined

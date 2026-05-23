import json
from pathlib import Path
import shlex

from tests.cross_matrix.run_release_regression_manifest import build_manifest_artifact
from tests.cross_matrix.release_regression_manifest import (
    CURRENT_POST_BUDGET_EDGE_ARTIFACTS,
    CURRENT_REGRESSION_SUITE_ARTIFACT,
    EXPECTED_CURRENT_API_SURFACE_CHECKS,
    EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS,
    EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS,
    EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS,
    EXPECTED_CURRENT_MCP_POLICY_CHECKS,
    EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS,
    EXPECTED_CURRENT_NATIVE_MTP_CHECKS,
    EXPECTED_CURRENT_OPEN_REQUIREMENTS,
    EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS,
    EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
    EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS,
    EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS,
    EXPECTED_CURRENT_TOOL_CALL_CHECKS,
    EXPECTED_CURRENT_VL_MEDIA_CHECKS,
    REQUIRED_RELEASE_DOMAINS,
    build_manifest,
    validate_current_proof_sweep_artifacts,
)


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


def test_release_regression_manifest_tracks_current_post_budget_edge_proof_sweep():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}

    for row_id, artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.items():
        assert artifact in rows[row_id]["artifacts"], row_id


def test_release_regression_manifest_validates_current_proof_sweep_artifacts(tmp_path):
    model_family_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    model_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    cache_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    parser_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    generation_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
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
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "pass"
    assert result["missing"] == []
    assert result["not_pass"] == []
    assert result["regression_suite"] == {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "pass",
        "failed_steps": [],
        "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
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
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": [],
        "failed_checks": [],
        "missing_expected_checks": [],
    }


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
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_current_proof_sweep_artifacts(tmp_path)

    assert result["status"] == "fail"
    assert result["missing"] == [artifacts[0]]
    assert result["not_pass"] == [{"artifact": artifacts[1], "status": "fail"}]


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
                    EXPECTED_CURRENT_OPEN_REQUIREMENTS[0],
                    "Server max output/context wiring regressed",
                ],
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
            EXPECTED_CURRENT_OPEN_REQUIREMENTS[0],
            "Server max output/context wiring regressed",
        ],
        "unexpected_open_requirements": ["Server max output/context wiring regressed"],
        "missing_expected_open_requirements": [EXPECTED_CURRENT_OPEN_REQUIREMENTS[1]],
    }


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
                            EXPECTED_CURRENT_OPEN_REQUIREMENTS[0],
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
    assert result["packaged_integrity_matrix"]["missing_expected_open_requirements"] == [
        EXPECTED_CURRENT_OPEN_REQUIREMENTS[1]
    ]


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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
                        },
                        "missing_markers": [],
                    }
                )
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
    api_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    reasoning_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    tool_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    mtp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    vl_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    mcp_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    max_output_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    packaged_artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
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
                json.dumps(
                    {
                        "status": "pass",
                        "checks": {
                            name: True for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS
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
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = build_manifest_artifact(tmp_path)

    assert artifact["current_proof_sweep"] == {
        "status": "pass",
        "missing": [],
        "not_pass": [],
        "regression_suite": {
            "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
            "status": "pass",
            "failed_steps": [],
            "open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "unexpected_open_requirements": [],
            "missing_expected_open_requirements": [],
        },
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
        "packaged_integrity_matrix": {
            "artifact": packaged_artifact,
            "status": "pass",
            "checks": {name: True for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS},
            "failed": [],
            "known_expected_release_gate_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS,
            "unexpected_open_requirements": [],
            "missing_expected_open_requirements": [],
            "failed_checks": [],
            "missing_expected_checks": [],
        },
    }


def test_release_regression_manifest_runner_can_require_current_proof_sweep(tmp_path):
    artifact = build_manifest_artifact(tmp_path, require_current_proof_sweep=True)

    assert artifact["status"] == "fail"
    assert artifact["current_proof_sweep"]["missing"]


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
    assert "current-generation-defaults-contract-20260521.json" in joined
    assert "generation_config.json" in joined
    assert "jang_config.json" in joined
    assert "hidden sampler forcing" in joined


def test_release_regression_manifest_tracks_reasoning_template_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["reasoning-template-no-think-tag-leak"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_reasoning_template_contract.py" in joined
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
    assert "current-api-surface-contract-20260522-stream-cache-detail.json" in joined
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
    assert "current-generation-defaults-contract-20260522-recheck-no-hidden-forcing.json" in generation_joined
    assert "bundle max_new_tokens" in generation_joined
    assert "without hidden sampler or repetition floors" in generation_joined
    assert "server default output cap is not a request ceiling" in generation_joined

    reasoning = rows["reasoning-template-no-think-tag-leak"]
    reasoning_joined = " ".join(reasoning["commands"] + reasoning["artifacts"] + reasoning["proves"])
    assert "current-reasoning-template-contract-20260522-recheck-parser-rails.json" in reasoning_joined
    assert "DSV4 requested reasoning rails are preserved" in reasoning_joined
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
    assert "current-tool-call-contract-20260521.json" in joined
    assert "Tool parser residue" in joined
    assert "DSV4" in joined
    assert "maxToolIterations" in joined
    assert "live DSV4 write_file DSML degradation is repaired schema-safely" in joined
    assert "current-tool-call-contract-20260522-dsv4-live-write-file-repair.json" in joined
    assert "current-dsv4-default-cache-tool-loop-poolon-materialized-parserfix-20260522/result.json" in joined


def test_release_regression_manifest_tracks_panel_cache_family_gating():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["panel-session-cache-settings-family-gating"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "cache_architecture"
    assert "run_noheavy_panel_settings_contract.py" in joined
    assert "current-panel-settings-contract-proof-20260522-launch-memory-warning.json" in joined
    assert "DSV4 pool quant" in joined
    assert "JANG/JANGTQ/MXFP" in joined
    assert "warning-only for lazy-mmap" in joined


def test_release_regression_manifest_tracks_mcp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["mcp-policy-ui-gateway"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_mcp_policy_contract.py" in joined
    assert "current-mcp-policy-contract-20260521.json" in joined
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
    assert "current-decode-speed-live-qwen27-jang4m-source-keepalloc-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-packaged-keepalloc-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-packaged-tahoe-dmg-20260522.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-source-bypass-fix-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json" in joined
    assert "current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json" in joined
    assert "MLX and MLX-metal wheel tags" in joined
    assert "selective live TurboQuant for Qwen attention KV layers" in joined
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
    assert "current-packaged-integrity-contract-20260521.json" in joined
    assert "Version triples" in joined
    assert "bundled Python hash parity" in joined
    assert "objective proof digest" in joined
    assert "objective-gate-enforced" in joined
    assert "verify-bundled" in joined


def test_release_regression_manifest_tracks_current_packaged_integrity_recheck():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["packaged-release-integrity"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json" in joined
    assert "clean JANG source path" in joined
    assert "bundled critical jang_tools files match source content" in joined
    assert "console-script shebangs are relocatable" in joined
    assert "fails only on the known DSV4 objective row" in joined


def test_release_regression_manifest_tracks_public_release_surface_preflight():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["public-release-surface-preflight"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert row["domain"] == "release_surface"
    assert "run_release_surface_contract.py" in joined
    assert "latest.json" in joined
    assert "PyPI" in joined
    assert "GitHub release" in joined
    assert "updater" in joined.lower()
    assert "post-release-updater" in joined
    assert "published updater state is complete" in joined
    assert "--live-public" in joined
    assert "mlx.studio/update/latest.json" in joined
    assert "PyPI files" in joined


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

    assert "current-production-family-audit-live-dsv4-jang-local-20260522-after-stream-cache-detail.json" in joined
    assert "identifier integrity" in joined


def test_release_regression_manifest_live_soak_does_not_overclaim_qwen_mtp():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["model-family-live-multiturn-soak"]
    command_text = " ".join(row["commands"]).lower()
    proves_text = " ".join(row["proves"]).lower()

    if "qwen mtp" in proves_text and "no-heavy" not in proves_text:
        assert "qwen" in command_text and "mtp" in command_text


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
    assert "ZAYA1-VL JANGTQ_K/JANGTQ2/JANGTQ4 qwen3 reasoning rails" in joined
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
    assert "current-cache-architecture-contract-20260521.json" in joined
    assert "DSV4 composite" in joined
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
    assert "DSV4 panel env mapping enables pool quant only from explicit DSV4 config" in joined
    assert "current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json" in joined
    assert "DSV4 timing probe covers prefix-cache replay and cold-store boundaries" in joined
    assert "Panel session launch builder preserves DSV4 default and diagnostic prefix-cache policy" in joined
    assert "Qwen3.6 hybrid and Mamba paged-cache forcing" in joined
    assert "regular KV stale saved false semantics" in joined


def test_release_regression_manifest_tracks_native_mtp_with_runner_artifact():
    manifest = build_manifest()
    rows = {row["id"]: row for row in manifest["rows"]}
    row = rows["native-mtp-d3-effect-policy"]
    joined = " ".join(row["commands"] + row["artifacts"] + row["proves"])

    assert "run_native_mtp_contract.py" in joined
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
    assert "current-native-mtp-speed-ab-qwen27-jang4m-mtp-20260523/result.json" in joined
    assert "same-artifact AR-vs-MTP" in joined
    assert "MTP decode is not the MLLM prefill bottleneck" in joined
    assert "prefill" in joined

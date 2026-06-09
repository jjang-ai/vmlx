# SPDX-License-Identifier: Apache-2.0
"""Contracts for the public app/runtime issue audit."""

from pathlib import Path


def test_public_app_issue_audit_tracks_open_app_runtime_issue_slices():
    from tests.cross_matrix import run_public_app_issue_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] in {"open", "pass"}
    assert set(audit["issues"]) == {
        "165",
        "166",
        "169",
        "180",
        "111",
        "115",
        "116",
        "117",
        "118",
        "119",
    }
    assert audit["issues"]["165"]["focused_source_slice"] in {"open", "pass"}
    assert audit["issues"]["165"]["checks"]["tool_call_contract_source_checks_pass"] is True
    assert isinstance(
        audit["issues"]["165"]["checks"]["tool_call_contract_passes"],
        bool,
    )
    assert audit["issues"]["165"]["checks"]["dsml_issue_165_regression_present"] is True
    assert isinstance(
        audit["issues"]["165"]["checks"]["installed_app_dsml_parser_hash_guarded"],
        bool,
    )
    assert audit["issues"]["165"]["release_clearance"] == (
        "mapped_to_dsv4_dsml_tool_call_arguments_guard"
    )
    assert audit["issues"]["166"]["focused_source_slice"] in {"open", "pass"}
    assert audit["issues"]["166"]["checks"]["source_gemma4_assistant_alias"] is True
    assert (
        audit["issues"]["166"]["checks"]["bundled_verify_gemma4_assistant_alias"]
        is True
    )
    assert isinstance(
        audit["issues"]["166"]["checks"]["installed_app_gemma4_assistant_alias"],
        bool,
    )
    assert audit["issues"]["166"]["release_clearance"] == (
        "mapped_to_gemma4_assistant_mlx_vlm_alias_guard"
    )
    assert audit["issues"]["169"]["focused_source_slice"] == "pass"
    assert audit["issues"]["169"]["checks"]["dual_public_dmg_flavors"] is True
    assert audit["issues"]["169"]["checks"]["compat_wheel_default"] is True
    assert (
        audit["issues"]["169"]["checks"][
            "installed_app_supported_runtime_flavor"
        ]
        is True
    )
    assert (
        audit["issues"]["169"]["checks"][
            "staged_sequoia_app_compat_runtime_flavor"
        ]
        is True
    )
    assert (
        audit["issues"]["169"]["checks"]["staged_tahoe_app_native_runtime_flavor"]
        is True
    )
    assert audit["issues"]["169"]["release_clearance"] == (
        "installed_supported_runtime_flavor_and_dual_staged_dmgs_guarded_packaging_still_gated"
    )
    assert audit["issues"]["117"]["focused_source_slice"] in {"open", "pass"}
    assert isinstance(
        audit["issues"]["117"]["checks"]["issue179_root_cause_audit_passes"],
        bool,
    )
    assert isinstance(
        audit["issues"]["117"]["checks"]["issue179_root_cause_audit_open"],
        bool,
    )
    assert audit["issues"]["117"]["checks"]["minimax_live_ui_artifacts_indexed"] is True
    assert audit["issues"]["117"]["release_clearance"] in {
        "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary",
        "open_minimax_k_issue179_reporter_parity_required",
    }
    assert audit["issues"]["180"]["focused_source_slice"] == "pass"
    assert audit["issues"]["180"]["checks"]["minimax_small_stricttools_real_ui_indexed"] is True
    assert audit["issues"]["180"]["checks"]["minimax_small_numeric_garbage_guarded"] is True
    assert audit["issues"]["180"]["release_clearance"] == (
        "mapped_to_minimax_small_real_ui_language_numeric_guard"
    )
    assert audit["issues"]["111"]["focused_source_slice"] in {"open", "pass"}
    assert audit["issues"]["111"]["checks"]["mistral_small4_wrapper_stays_mllm"] is True
    assert audit["issues"]["111"]["checks"]["mistral_small4_parser_metadata_preserved"] is True
    assert isinstance(
        audit["issues"]["111"]["checks"]["installed_app_mllm_hash_guarded"],
        bool,
    )
    assert audit["issues"]["111"]["release_clearance"] == (
        "mapped_to_mistral_small4_vlm_wrapper_detection_guard"
    )
    assert audit["issues"]["116"]["focused_source_slice"] == "pass"
    assert audit["issues"]["116"]["checks"]["reasoning_template_contract_passes"] is True
    assert audit["issues"]["116"]["checks"]["explicit_thinking_off_request_wired"] is True
    assert audit["issues"]["116"]["checks"]["panel_thinking_off_control_present"] is True
    assert audit["issues"]["116"]["checks"]["packaged_renderer_thinking_controls_present"] is True
    assert audit["issues"]["116"]["release_clearance"] == (
        "mapped_to_thinking_off_ui_api_request_guard"
    )
    assert audit["issues"]["115"]["focused_source_slice"] in {"open", "pass"}
    assert audit["issues"]["115"]["checks"]["gemma4_installed_speed_risk_tracked"] is True
    assert isinstance(
        audit["issues"]["115"]["checks"]["gemma4_current_installed_ui_speed_gate_passes"],
        bool,
    )
    assert isinstance(
        audit["issues"]["115"]["checks"]["gemma4_cold_wall_includes_ttft_tracked"],
        bool,
    )
    assert audit["issues"]["115"]["checks"]["qwen36_speed_review_tracked"] is True
    assert isinstance(
        audit["issues"]["115"]["checks"]["qwen35_installed_app_speed_gate_passes"],
        bool,
    )
    assert audit["issues"]["115"]["release_clearance"] == (
        "mapped_to_current_installed_app_gemma_qwen_speed_gate"
    )
    assert audit["issues"]["118"]["focused_source_slice"] == "pass"
    assert audit["issues"]["118"]["checks"]["download_worker_clears_raw_endpoint"] is True
    assert audit["issues"]["118"]["checks"]["api_retries_without_stale_auth"] is True
    assert (
        audit["issues"]["118"]["checks"]["installed_app_download_fallback_guarded"]
        is True
    )
    assert audit["issues"]["118"]["release_clearance"] == (
        "installed_gui_download_endpoint_and_stale_auth_fallback_guarded"
    )
    assert audit["issues"]["119"]["focused_source_slice"] in {"open", "pass"}
    assert isinstance(
        audit["issues"]["119"]["checks"]["gemma26_memory_stress_artifact_present"],
        bool,
    )
    assert isinstance(
        audit["issues"]["119"]["checks"][
            "gemma26_memory_stress_native_cache_health"
        ],
        bool,
    )
    assert isinstance(
        audit["issues"]["119"]["checks"][
            "gemma26_memory_stress_mixed_swa_cache_hits"
        ],
        bool,
    )
    assert audit["issues"]["119"]["checks"]["gemma26_real_ui_artifacts_indexed"] is True
    assert audit["issues"]["119"]["release_clearance"] == (
        "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
    )


def test_public_app_issue_audit_uses_current_manifest_artifact():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_public_app_issue_audit as gate

    assert str(gate.DEFAULT_OUT) == manifest.CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    assert str(gate.ISSUE179_ROOT_CAUSE_AUDIT) == (
        manifest.CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    )
    assert str(gate.ISSUE179_RESPONSES_CANCEL_PROOF) == (
        "build/current-issue179-minimax-k-responses-cancel-probe-20260606-live-refresh.json"
    )
    assert str(gate.ISSUE179_RESPONSES_CANCEL_PREFLIGHT) == (
        manifest.CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    )
    assert str(gate.PACKAGED_INTEGRITY_CONTRACT) == (
        manifest.CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    )


def test_public_app_issue117_open_boundary_uses_preflight_without_live_cancel_probe(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_public_app_issue_audit as gate

    root = tmp_path
    root_cause = Path("build/issue179-root-cause.json")
    live_cancel = Path("build/issue179-live-cancel.json")
    preflight = Path("build/issue179-memory-preflight.json")
    monkeypatch.setattr(gate, "ISSUE179_ROOT_CAUSE_AUDIT", root_cause)
    monkeypatch.setattr(gate, "ISSUE179_RESPONSES_CANCEL_PROOF", live_cancel)
    monkeypatch.setattr(gate, "ISSUE179_RESPONSES_CANCEL_PREFLIGHT", preflight)

    (root / root_cause).parent.mkdir(parents=True, exist_ok=True)
    (root / root_cause).write_text('{"status": "open"}', encoding="utf-8")
    (root / preflight).write_text(
        '{"status": "ready_to_launch", "did_not_launch": true,'
        '"launch_decision": "launch_allowed"}',
        encoding="utf-8",
    )
    (root / "tests/cross_matrix").mkdir(parents=True, exist_ok=True)
    (root / "tests/cross_matrix/release_regression_manifest.py").write_text(
        "current-real-ui-live-model-minimax-m27-small",
        encoding="utf-8",
    )
    (root / "tests/cross_matrix/run_parser_registry_contract.py").write_text(
        "passes MiniMax through the registered minimax_m2 reasoning parser\n"
        "uses the registered MiniMax reasoning parser even when bundle sidecars say qwen3",
        encoding="utf-8",
    )
    note = (
        root
        / "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json"
    )
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        '{"modelName":"MiniMax-M2.7-Small-JANGTQ",'
        '"provenSurfaces":["cache_hit_telemetry","parser_leak_check","language_leak_check"]}',
        encoding="utf-8",
    )

    audit = gate.build_audit(root)

    assert not (root / live_cancel).exists()
    assert "117" in audit["focused_open"]
    assert "117" not in audit["focused_failures"]
    assert audit["issues"]["117"]["focused_source_slice"] == "open"
    assert audit["issues"]["117"]["checks"]["issue179_cancel_probe_present"] is False
    assert (
        audit["issues"]["117"]["checks"][
            "issue179_cancel_probe_memory_preflight_present"
        ]
        is True
    )
    assert audit["issues"]["117"]["release_clearance"] == (
        "open_minimax_k_issue179_reporter_parity_required"
    )


def test_public_app_issue_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_public_app_issue_audit as gate

    out = tmp_path / "public-app-issue-audit.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert f'"status": "{audit["status"]}"' in out.read_text(encoding="utf-8")
    assert audit["artifact"] == str(out)


def test_public_app_issue_audit_cli_accepts_cleared_issue179_boundary(
    monkeypatch,
    tmp_path,
):
    from tests.cross_matrix import run_public_app_issue_audit as gate

    out = tmp_path / "public-app-issue-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_public_app_issue_audit.py",
            "--out",
            str(out),
        ],
    )

    assert gate.main() == 0
    assert any(
        marker in out.read_text(encoding="utf-8")
        for marker in ('"status": "pass"', '"status": "open"')
    )

# SPDX-License-Identifier: Apache-2.0
"""Contracts for the public app/runtime issue audit."""

from pathlib import Path


def test_public_app_issue_audit_tracks_open_app_runtime_issue_slices():
    from tests.cross_matrix import run_public_app_issue_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] == "pass"
    assert set(audit["issues"]) == {
        "165",
        "169",
        "180",
        "111",
        "116",
        "117",
        "118",
        "119",
    }
    assert audit["issues"]["165"]["focused_source_slice"] == "pass"
    assert audit["issues"]["165"]["checks"]["tool_call_contract_passes"] is True
    assert audit["issues"]["165"]["checks"]["dsml_issue_165_regression_present"] is True
    assert audit["issues"]["165"]["checks"]["installed_app_dsml_parser_hash_guarded"] is True
    assert audit["issues"]["165"]["release_clearance"] == (
        "mapped_to_dsv4_dsml_tool_call_arguments_guard"
    )
    assert audit["issues"]["169"]["focused_source_slice"] == "pass"
    assert audit["issues"]["169"]["checks"]["dual_public_dmg_flavors"] is True
    assert audit["issues"]["169"]["checks"]["compat_wheel_default"] is True
    assert (
        audit["issues"]["169"]["checks"][
            "installed_app_sequoia_compat_runtime_flavor"
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
        "installed_and_staged_sequoia_compat_runtime_flavor_guarded_packaging_still_gated"
    )
    assert audit["issues"]["117"]["focused_source_slice"] == "pass"
    assert audit["issues"]["117"]["checks"]["issue179_root_cause_audit_passes"] is True
    assert audit["issues"]["117"]["checks"]["minimax_live_ui_artifacts_indexed"] is True
    assert audit["issues"]["117"]["release_clearance"] == (
        "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary"
    )
    assert audit["issues"]["180"]["focused_source_slice"] == "pass"
    assert audit["issues"]["180"]["checks"]["minimax_small_stricttools_real_ui_indexed"] is True
    assert audit["issues"]["180"]["checks"]["minimax_small_numeric_garbage_guarded"] is True
    assert audit["issues"]["180"]["release_clearance"] == (
        "mapped_to_minimax_small_real_ui_language_numeric_guard"
    )
    assert audit["issues"]["111"]["focused_source_slice"] == "pass"
    assert audit["issues"]["111"]["checks"]["mistral_small4_wrapper_stays_mllm"] is True
    assert audit["issues"]["111"]["checks"]["mistral_small4_parser_metadata_preserved"] is True
    assert audit["issues"]["111"]["checks"]["installed_app_mllm_hash_guarded"] is True
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
    assert audit["issues"]["119"]["focused_source_slice"] == "pass"
    assert audit["issues"]["119"]["checks"]["gemma26_memory_stress_artifact_present"] is True
    assert audit["issues"]["119"]["checks"]["gemma26_real_ui_artifacts_indexed"] is True
    assert audit["issues"]["119"]["release_clearance"] == (
        "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
    )


def test_public_app_issue_audit_uses_current_manifest_artifact():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_public_app_issue_audit as gate

    assert str(gate.DEFAULT_OUT) == manifest.CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT


def test_public_app_issue_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_public_app_issue_audit as gate

    out = tmp_path / "public-app-issue-audit.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert f'"status": "{audit["status"]}"' in out.read_text(encoding="utf-8")
    assert audit["artifact"] == str(out)

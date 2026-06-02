# SPDX-License-Identifier: Apache-2.0
"""Contracts for the no-heavy #175-#179 release-boundary audit."""

from pathlib import Path


def test_issue175_179_audit_preserves_open_release_boundaries():
    from tests.cross_matrix import run_issue175_179_release_boundary_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] in {"open", "pass"}
    assert set(audit["issues"]) == {"175", "176", "177", "178", "179"}
    assert audit["issues"]["175"]["focused_source_slice"] == "pass"
    assert audit["issues"]["175"]["checks"]["installed_app_memory_clear_runtime_proven"] is True
    assert audit["issues"]["175"]["release_clearance"] == (
        "installed_app_memory_clear_runtime_live_stress_proven"
    )
    assert audit["issues"]["176"]["focused_source_slice"] == "pass"
    assert audit["issues"]["176"]["checks"][
        "l2_readable_write_through_regression_test_present"
    ] is True
    assert audit["issues"]["176"]["checks"]["installed_app_promoted_block_cleanup_proven"] is True
    assert audit["issues"]["176"]["release_clearance"] == (
        "installed_app_promoted_block_cleanup_live_pressure_proven"
    )
    assert audit["issues"]["177"]["focused_source_slice"] == "pass"
    assert audit["issues"]["177"]["checks"][
        "worker_timing_regression_test_present"
    ] is True
    assert audit["issues"]["177"]["checks"]["installed_app_cache_selection_telemetry_proven"] is True
    assert audit["issues"]["177"]["checks"]["installed_live_ttft_and_cold_paged_tq_proven"] is True
    assert audit["issues"]["177"]["release_clearance"] == (
        "installed_app_cache_selection_live_ttft_proven"
    )
    assert audit["issues"]["178"]["focused_source_slice"] == "pass"
    assert audit["issues"]["178"]["checks"]["text_lora_flags_rejected"] is True
    assert audit["issues"]["178"]["release_clearance"] in {
        "packaged_app_lora_surface_proven",
        "open_packaged_image_lora_proof_required",
    }
    assert audit["issues"]["179"]["focused_source_slice"] == "open"
    assert audit["issues"]["179"]["checks"]["local_installed_live_cancel_probe_proven"] is True
    assert audit["issues"]["179"]["checks"]["reporter_parity_proven"] is False
    assert audit["issues"]["179"]["checks"]["local_reporter_prompt_reproduction_clean"] is True
    assert audit["issues"]["179"]["release_clearance"] == (
        "open_reporter_parity_required"
    )
    assert audit["required_live_proof_axes"] == [
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
    ]
    assert "Cross-family live multi-turn smoke matrix is release-cleared" not in audit["open_release_rows"]
    assert "Real Electron UI unblocked non-MiMo live model matrix is proven" not in audit["open_release_rows"]
    assert "Real Electron UI cross-family live model matrix is release-cleared" in audit["open_release_rows"]
    assert "DSV4 long-output/code/file-generation quality is release-cleared" in audit["open_release_rows"]
    assert "Issue #175-#178 focused installed/live runtime boundaries are proven" in audit[
        "release_boundary"
    ]
    assert "Issue #179 remains open" in audit["release_boundary"]
    assert "Issue #175-#179 focused installed/live runtime boundaries are proven" not in audit[
        "release_boundary"
    ]


def test_issue175_179_audit_uses_current_manifest_artifacts():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_issue175_179_release_boundary_audit as gate

    assert str(gate.DEFAULT_OUT) == manifest.CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    assert str(gate.INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT) == (
        manifest.CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT
    )
    assert str(gate.ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT) == (
        manifest.CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT
    )
    assert str(gate.ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT) == (
        manifest.CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT
    )
    assert str(gate.ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT) == (
        manifest.CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    )


def test_issue175_179_audit_uses_june_local_refresh_issue175_177_inputs():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_issue175_179_release_boundary_audit as gate

    installed = str(gate.ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT)
    live = str(gate.ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT)

    assert installed == manifest.CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT
    assert live == manifest.CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT
    assert installed.endswith("20260601-local-refresh.json")
    assert live.endswith("20260601-local-refresh.json")
    assert "20260527" not in installed
    assert "20260527" not in live


def test_issue175_179_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_issue175_179_release_boundary_audit as gate

    out = tmp_path / "issue-boundary.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert f'"status": "{audit["status"]}"' in out.read_text(encoding="utf-8")
    assert audit["artifact"] == str(out)

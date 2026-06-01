# SPDX-License-Identifier: Apache-2.0
"""Contracts for the no-heavy #181-#183 runtime issue audit."""

from pathlib import Path


def test_issue181_183_audit_proves_recent_runtime_issue_slices():
    from tests.cross_matrix import run_issue181_183_runtime_audit as gate

    audit = gate.build_audit(Path("."))

    assert audit["status"] == "pass"
    assert set(audit["issues"]) == {"181", "182", "183"}
    assert audit["issues"]["181"]["focused_source_slice"] == "pass"
    assert audit["issues"]["181"]["checks"]["mpp_auto_disabled_for_mxtq"] is True
    assert audit["issues"]["181"]["checks"]["explicit_mpp_on_still_allowed"] is True
    assert (
        audit["issues"]["181"]["checks"]["installed_app_mpp_auto_policy_disables_mxtq"]
        is True
    )
    assert audit["issues"]["181"]["release_clearance"] == (
        "installed_mpp_auto_policy_guarded"
    )
    assert audit["issues"]["182"]["focused_source_slice"] == "pass"
    assert audit["issues"]["182"]["checks"]["normal_vlm_patch_embed_transpose"] is True
    assert audit["issues"]["182"]["checks"]["native_mtp_patch_embed_transpose"] is True
    assert audit["issues"]["182"]["checks"]["bundled_hash_gate_covers_runtime"] is True
    assert (
        audit["issues"]["182"]["checks"]["installed_app_qwen_vl_patch_embed_layout"]
        is True
    )
    assert audit["issues"]["182"]["release_clearance"] == (
        "installed_qwen_vl_patch_embed_layout_guarded"
    )
    assert audit["issues"]["183"]["focused_source_slice"] == "pass"
    assert audit["issues"]["183"]["checks"]["minicpm_v46_registry_remap"] is True
    assert audit["issues"]["183"]["checks"]["minicpm_v46_prompt_config_remap"] is True
    assert audit["issues"]["183"]["checks"]["bundled_import_gate_covers_runtime"] is True
    assert audit["issues"]["183"]["checks"]["installed_app_minicpm_v46_runtime_remap"] is True
    assert audit["issues"]["183"]["release_clearance"] == (
        "source_and_packaged_minicpm_v46_load_guarded"
    )


def test_issue181_183_audit_uses_current_manifest_artifact():
    from tests.cross_matrix import release_regression_manifest as manifest
    from tests.cross_matrix import run_issue181_183_runtime_audit as gate

    assert str(gate.DEFAULT_OUT) == manifest.CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT


def test_issue181_183_audit_writes_json_artifact(tmp_path):
    from tests.cross_matrix import run_issue181_183_runtime_audit as gate

    out = tmp_path / "issue181-183.json"
    audit = gate.write_audit(Path("."), out)

    assert out.exists()
    assert f'"status": "{audit["status"]}"' in out.read_text(encoding="utf-8")
    assert audit["artifact"] == str(out)

from pathlib import Path

from tests.cross_matrix import run_signed_checkpoint_dmg_audit as audit


def test_signed_checkpoint_audit_records_current_and_stale_release_boundaries(tmp_path):
    out = tmp_path / "signed-checkpoint.json"
    result = audit.build_audit(root=Path("."), out=out)

    assert result["status"] in {"pass", "open"}
    assert result["head"] == audit.git_head(Path("."))
    assert result["current_installed_app"]["path"] == "/Applications/vMLX.app"
    assert "signature_is_adhoc" in result["current_installed_app"]
    assert result["fresh_signing_probe"]["identity"].startswith("Developer ID Application:")
    assert "vmlx-build.keychain-db" in result["notarization"]["required_keychain"]
    if result["fresh_signing_probe"]["status"] == "pass":
        assert "unlock_vmlx_build_keychain" not in result["required_next_steps"]
        assert "restore_codesign_partition_list" not in result["required_next_steps"]
        assert "rerun_fresh_developer_id_signing_probe" not in result["required_next_steps"]
    else:
        assert result["required_next_steps"][:3] == [
            "unlock_vmlx_build_keychain",
            "restore_codesign_partition_list",
            "rerun_fresh_developer_id_signing_probe",
        ]
    assert "rebuild_current_source_dmg_flavors" in result["required_next_steps"]
    assert result["required_next_steps"][-1] == "notarize_staple_and_verify_current_dmgs"
    assert out.exists()


def test_signed_checkpoint_audit_next_steps_skip_signing_repair_when_access_is_green():
    assert audit.required_next_steps(
        signing={"status": "pass"},
        notary={"status": "pass"},
        existing_dmgs_current_for_head=False,
    ) == [
        "rebuild_current_source_dmg_flavors",
        "notarize_staple_and_verify_current_dmgs",
    ]


def test_signed_checkpoint_audit_next_steps_include_repair_when_signing_is_blocked():
    assert audit.required_next_steps(
        signing={"status": "blocked"},
        notary={"status": "blocked"},
        existing_dmgs_current_for_head=False,
    ) == [
        "unlock_vmlx_build_keychain",
        "restore_codesign_partition_list",
        "rerun_fresh_developer_id_signing_probe",
        "rebuild_current_source_dmg_flavors",
        "restore_vmlx_notary_keychain_profile",
        "notarize_staple_and_verify_current_dmgs",
    ]

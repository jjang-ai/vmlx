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
    assert result["required_next_steps"] == [
        "unlock_vmlx_build_keychain",
        "restore_codesign_partition_list",
        "rerun_fresh_developer_id_signing_probe",
        "rebuild_current_source_dmg_flavors",
        "notarize_staple_and_verify_current_dmgs",
    ]
    assert out.exists()

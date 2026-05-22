import json
from pathlib import Path


def test_release_surface_contract_allows_source_ahead_of_local_updater(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    (tmp_path / "pyproject.toml").write_text('version = "1.5.47"\n', encoding="utf-8")
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "package.json").write_text(
        json.dumps({"version": "1.5.47"}), encoding="utf-8"
    )
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "version": "1.5.46",
                "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v1.5.46/vMLX-1.5.46-sequoia-arm64.dmg",
                "sha256": "a" * 64,
                "notes": "vMLX 1.5.46",
            }
        ),
        encoding="utf-8",
    )

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "pass"
    assert artifact["checks"]["source_version_consistent"] is True
    assert artifact["checks"]["local_updater_not_ahead_of_source"] is True
    assert artifact["checks"]["staged_source_version_not_public"] is True


def test_release_surface_contract_rejects_updater_ahead_of_source(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    (tmp_path / "pyproject.toml").write_text('version = "1.5.47"\n', encoding="utf-8")
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "package.json").write_text(
        json.dumps({"version": "1.5.47"}), encoding="utf-8"
    )
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "version": "1.5.48",
                "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v1.5.48/vMLX-1.5.48-sequoia-arm64.dmg",
                "sha256": "b" * 64,
                "notes": "vMLX 1.5.48",
            }
        ),
        encoding="utf-8",
    )

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["checks"]["local_updater_not_ahead_of_source"] is False


def test_release_surface_contract_requires_complete_local_updater_when_bumped(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    (tmp_path / "pyproject.toml").write_text('version = "1.5.47"\n', encoding="utf-8")
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "package.json").write_text(
        json.dumps({"version": "1.5.47"}), encoding="utf-8"
    )
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "version": "1.5.47",
                "url": "https://example.com/vMLX-1.5.46.dmg",
                "sha256": "not-a-real-sha",
                "notes": "old notes",
            }
        ),
        encoding="utf-8",
    )

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["checks"]["local_updater_url_matches_version"] is False
    assert artifact["checks"]["local_updater_sha256_valid"] is False
    assert artifact["checks"]["local_updater_notes_match_version"] is False

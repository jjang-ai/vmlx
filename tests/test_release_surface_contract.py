import json
from pathlib import Path


def test_release_surface_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_release_surface_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-release-surface-contract-20260602-v154-live-public-after-site-fix.json"
    )


def _write_release_root(tmp_path: Path, version: str = "1.5.48") -> dict[str, str]:
    (tmp_path / "pyproject.toml").write_text(
        f'version = "{version}"\n', encoding="utf-8"
    )
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "package.json").write_text(
        json.dumps({"version": version}), encoding="utf-8"
    )
    latest = {
        "version": version,
        "url": f"https://github.com/jjang-ai/mlxstudio/releases/download/v{version}/vMLX-{version}-sequoia-arm64.dmg",
        "sha256": "a" * 64,
        "downloads": {
            "sequoia": {
                "url": f"https://github.com/jjang-ai/mlxstudio/releases/download/v{version}/vMLX-{version}-sequoia-arm64.dmg",
                "sha256": "a" * 64,
            },
            "tahoe": {
                "url": f"https://github.com/jjang-ai/mlxstudio/releases/download/v{version}/vMLX-{version}-tahoe-arm64.dmg",
                "sha256": "b" * 64,
            },
        },
        "notes": f"vMLX {version}\n\nRelease notes.",
    }
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    return latest


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


def test_release_surface_contract_allows_complete_post_release_updater(tmp_path):
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
                "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v1.5.47/vMLX-1.5.47-sequoia-arm64.dmg",
                "sha256": "c" * 64,
                "downloads": {
                    "sequoia": {
                        "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v1.5.47/vMLX-1.5.47-sequoia-arm64.dmg",
                        "sha256": "c" * 64,
                    },
                    "tahoe": {
                        "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v1.5.47/vMLX-1.5.47-tahoe-arm64.dmg",
                        "sha256": "d" * 64,
                    },
                },
                "notes": "vMLX 1.5.47\n\nKnown follow-up: DSV4 quality row.",
            }
        ),
        encoding="utf-8",
    )

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "pass"
    assert artifact["checks"]["local_updater_not_ahead_of_source"] is True
    assert artifact["checks"]["local_updater_release_state_valid"] is True


def test_release_surface_contract_requires_platform_downloads_when_updater_is_public(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)
    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "pass"
    assert artifact["checks"]["local_updater_platform_downloads_valid"] is True
    assert artifact["local_latest"]["downloads"]["tahoe"]["url"] == latest["downloads"]["tahoe"]["url"]


def test_release_surface_contract_rejects_missing_tahoe_download_for_public_updater(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)
    latest["downloads"].pop("tahoe")
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")

    artifact = gate.build_artifact(tmp_path)

    assert artifact["status"] == "fail"
    assert artifact["checks"]["local_updater_platform_downloads_valid"] is False


def test_release_surface_live_public_checks_detect_stale_site_updater(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url:
            return latest
        if "mlx.studio/update/latest.json" in url:
            return {**latest, "version": "1.5.46"}
        if "pypi.org" in url:
            return {
                "info": {"version": "1.5.48"},
                "urls": [
                    {"filename": "vmlx-1.5.48-py3-none-any.whl"},
                    {"filename": "vmlx-1.5.48.tar.gz"},
                ],
            }
        if "api.github.com" in url:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "vMLX-1.5.48-sequoia-arm64.dmg",
                        "digest": "sha256:" + latest["sha256"],
                    }
                ],
            }
        raise AssertionError(url)

    artifact = gate.build_artifact(tmp_path, live_public=True, fetch_json=fetch_json)

    assert artifact["status"] == "fail"
    assert artifact["checks"]["public_raw_updater_matches_local"] is True
    assert artifact["checks"]["public_site_updater_matches_local"] is False


def test_release_surface_live_public_checks_require_site_updater_no_store_headers(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url or "mlx.studio/update/latest.json" in url:
            return latest
        if "pypi.org" in url:
            return {
                "info": {"version": "1.5.48"},
                "urls": [
                    {"filename": "vmlx-1.5.48-py3-none-any.whl"},
                    {"filename": "vmlx-1.5.48.tar.gz"},
                ],
            }
        if "api.github.com" in url:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "vMLX-1.5.48-sequoia-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["sequoia"]["sha256"],
                    },
                    {
                        "name": "vMLX-1.5.48-tahoe-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["tahoe"]["sha256"],
                    },
                ],
            }
        raise AssertionError(url)

    artifact = gate.build_artifact(
        tmp_path,
        live_public=True,
        fetch_json=fetch_json,
        fetch_headers=lambda url: {"cache-control": "public, max-age=31536000"},
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["public_site_updater_cache_headers_safe"] is False


def test_release_surface_live_public_checks_require_pypi_files(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url or "mlx.studio/update/latest.json" in url:
            return latest
        if "pypi.org" in url:
            return {"info": {"version": "1.5.48"}, "urls": []}
        if "releases/tags" in url:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "vMLX-1.5.48-sequoia-arm64.dmg",
                        "digest": "sha256:" + latest["sha256"],
                    },
                    {
                        "name": "vMLX-1.5.48-tahoe-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["tahoe"]["sha256"],
                    },
                ],
            }
        if "git/ref/tags" in url:
            return {"object": {"sha": "current-source-head"}}
        raise AssertionError(url)

    artifact = gate.build_artifact(
        tmp_path,
        live_public=True,
        fetch_json=fetch_json,
        fetch_headers=lambda url: {
            "cache-control": "no-cache, no-store, must-revalidate",
            "pragma": "no-cache",
            "expires": "0",
        },
        current_revision="current-source-head",
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["public_pypi_has_release_files"] is False
    assert artifact["checks"]["staged_source_version_not_public"] is False
    assert artifact["status_failed_checks"] == ["public_pypi_has_release_files"]
    assert "staged_source_version_not_public" in artifact["informational_false_checks"]


def test_release_surface_live_public_checks_require_github_release_asset(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url or "mlx.studio/update/latest.json" in url:
            return latest
        if "pypi.org" in url:
            return {
                "info": {"version": "1.5.48"},
                "urls": [
                    {"filename": "vmlx-1.5.48-py3-none-any.whl"},
                    {"filename": "vmlx-1.5.48.tar.gz"},
                ],
            }
        if "api.github.com" in url:
            return {"draft": False, "prerelease": False, "assets": []}
        raise AssertionError(url)

    artifact = gate.build_artifact(tmp_path, live_public=True, fetch_json=fetch_json)

    assert artifact["status"] == "fail"
    assert artifact["checks"]["public_github_release_has_updater_asset"] is False


def test_release_surface_live_public_checks_require_all_manifest_download_assets(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url or "mlx.studio/update/latest.json" in url:
            return latest
        if "pypi.org" in url:
            return {
                "info": {"version": "1.5.48"},
                "urls": [
                    {"filename": "vmlx-1.5.48-py3-none-any.whl"},
                    {"filename": "vmlx-1.5.48.tar.gz"},
                ],
            }
        if "api.github.com" in url:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "vMLX-1.5.48-sequoia-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["sequoia"]["sha256"],
                    }
                ],
            }
        raise AssertionError(url)

    artifact = gate.build_artifact(tmp_path, live_public=True, fetch_json=fetch_json)

    assert artifact["status"] == "fail"
    assert artifact["checks"]["public_github_release_has_all_manifest_download_assets"] is False


def test_release_surface_live_public_checks_require_release_tag_to_match_source_head(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url or "mlx.studio/update/latest.json" in url:
            return latest
        if "pypi.org" in url:
            return {
                "info": {"version": "1.5.48"},
                "urls": [
                    {"filename": "vmlx-1.5.48-py3-none-any.whl"},
                    {"filename": "vmlx-1.5.48.tar.gz"},
                ],
            }
        if "releases/tags" in url:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "vMLX-1.5.48-sequoia-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["sequoia"]["sha256"],
                    },
                    {
                        "name": "vMLX-1.5.48-tahoe-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["tahoe"]["sha256"],
                    },
                ],
            }
        if "git/ref/tags" in url:
            return {"object": {"sha": "old-release-commit"}}
        raise AssertionError(url)

    artifact = gate.build_artifact(
        tmp_path,
        live_public=True,
        fetch_json=fetch_json,
        current_revision="current-source-head",
    )

    assert artifact["status"] == "fail"
    assert artifact["checks"]["public_github_source_release_tag_matches_source_head"] is False
    assert artifact["public_surfaces"]["github_source_release_tag"]["sha"] == "old-release-commit"


def test_release_surface_live_public_checks_accept_annotated_release_tag(tmp_path):
    from tests.cross_matrix import run_release_surface_contract as gate

    latest = _write_release_root(tmp_path)

    def fetch_json(url: str):
        if "raw.githubusercontent.com" in url or "mlx.studio/update/latest.json" in url:
            return latest
        if "pypi.org" in url:
            return {
                "info": {"version": "1.5.48"},
                "urls": [
                    {"filename": "vmlx-1.5.48-py3-none-any.whl"},
                    {"filename": "vmlx-1.5.48.tar.gz"},
                ],
            }
        if "releases/tags" in url:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "vMLX-1.5.48-sequoia-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["sequoia"]["sha256"],
                    },
                    {
                        "name": "vMLX-1.5.48-tahoe-arm64.dmg",
                        "digest": "sha256:" + latest["downloads"]["tahoe"]["sha256"],
                    },
                ],
            }
        if "git/ref/tags" in url:
            return {"object": {"sha": "annotated-tag-object", "type": "tag"}}
        if "git/tags/annotated-tag-object" in url:
            return {"object": {"sha": "current-source-head", "type": "commit"}}
        raise AssertionError(url)

    artifact = gate.build_artifact(
        tmp_path,
        live_public=True,
        fetch_json=fetch_json,
        fetch_headers=lambda url: {
            "cache-control": "no-cache, no-store, must-revalidate",
            "pragma": "no-cache",
            "expires": "0",
        },
        current_revision="current-source-head",
    )

    assert artifact["status"] == "pass"
    assert artifact["checks"]["public_github_source_release_tag_matches_source_head"] is True
    tag = artifact["public_surfaces"]["github_source_release_tag"]
    assert tag["type"] == "tag"
    assert tag["resolved_sha"] == "current-source-head"
    assert tag["resolved_type"] == "commit"

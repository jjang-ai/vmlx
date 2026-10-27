import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/release_metadata.py"
SITE_SCRIPT = ROOT / ".github/scripts/deploy_website_release.py"


def _load_metadata_module():
    spec = importlib.util.spec_from_file_location("release_metadata", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _write_candidate(tmp_path: Path, version: str = "9.8.7") -> tuple[Path, Path]:
    dist = tmp_path / "dmgs"
    dist.mkdir()
    for flavor in ("sequoia", "tahoe"):
        dmg = dist / f"vMLX-{version}-{flavor}-arm64.dmg"
        dmg.write_bytes((flavor * 100).encode())
        (dist / f"{dmg.name}.blockmap").write_text(f"{flavor}-blockmap")
    jang = tmp_path / "jang-tools"
    jang.mkdir()
    (jang / "pyproject.toml").write_text(
        '[project]\nname = "jang"\nversion = "2.5.99"\n', encoding="utf-8"
    )
    return dist, jang


def test_release_metadata_round_trip_and_tahoe_default(tmp_path):
    module = _load_metadata_module()
    dist, jang = _write_candidate(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Jinho Jang")
    _git(repo, "config", "user.email", "eric@jangq.ai")
    (repo / "one").write_text("one")
    _git(repo, "add", "one")
    _git(repo, "commit", "-m", "First release change")
    _git(repo, "tag", "v9.8.6")
    (repo / "two").write_text("two")
    _git(repo, "add", "two")
    _git(repo, "commit", "-m", "Second release change")
    source_sha = _git(repo, "rev-parse", "HEAD")
    source_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    jang_sha = "b" * 40
    metadata = tmp_path / "release-info.json"
    notes = tmp_path / "release-notes.md"
    python_dist = tmp_path / "python-dist"
    python_dist.mkdir()
    (python_dist / "vmlx-9.8.7-py3-none-any.whl").write_bytes(b"wheel")
    (python_dist / "vmlx-9.8.7.tar.gz").write_bytes(b"sdist")

    create_args = type(
        "Args",
        (),
        {
            "version": "9.8.7",
            "source_sha": source_sha,
            "source_tree": source_tree,
            "jang_sha": jang_sha,
            "final_notary_manifest_sha256": "c" * 64,
            "dist": dist,
            "python_dist": python_dist,
            "jang_tools": jang,
            "root": repo,
            "previous_tag": "v9.8.6",
            "output": metadata,
            "notes_output": notes,
        },
    )()
    module.create_metadata(create_args)
    module.verify_metadata(
        type(
            "Args",
            (),
            {
                "metadata": metadata,
                "dist": dist,
                "python_dist": python_dist,
                "version": "9.8.7",
                "source_sha": source_sha,
                "jang_sha": jang_sha,
            },
        )()
    )
    updater = tmp_path / "latest.json"
    module.write_updater(
        type("Args", (), {"metadata": metadata, "notes": notes, "output": updater})()
    )

    release_info = json.loads(metadata.read_text())
    latest = json.loads(updater.read_text())
    assert release_info["source"] == {
        "repository": "jjang-ai/vmlx",
        "commit": source_sha,
        "tree": source_tree,
    }
    assert release_info["jangq"]["commit"] == jang_sha
    assert release_info["python_distributions"]["wheel"]["filename"] == (
        "vmlx-9.8.7-py3-none-any.whl"
    )
    assert "Second release change" in notes.read_text()
    assert latest["url"] == latest["downloads"]["tahoe"]["url"]
    assert latest["sha256"] == latest["downloads"]["tahoe"]["sha256"]
    assert latest["downloads"]["sequoia"]["url"].endswith(
        "/v9.8.7/vMLX-9.8.7-sequoia-arm64.dmg"
    )

    assert module.classify_pypi_release(release_info, None) == "missing"
    pypi_payload = {
        "urls": [
            {
                "filename": record["filename"],
                "size": record["bytes"],
                "digests": {"sha256": record["sha256"]},
            }
            for record in release_info["python_distributions"].values()
        ]
    }
    assert module.classify_pypi_release(release_info, pypi_payload) == "exact"
    pypi_payload["urls"][0]["size"] += 1
    with pytest.raises(SystemExit, match="different artifact set"):
        module.classify_pypi_release(release_info, pypi_payload)


def test_website_deployer_backs_up_and_updates_both_flavors(tmp_path):
    site = tmp_path / "site"
    (site / "update").mkdir(parents=True)
    (site / "download").mkdir()
    old_latest = {"version": "9.8.6"}
    (site / "update/latest.json").write_text(json.dumps(old_latest))
    old_html = """\
    <script>{"softwareVersion": "9.8.6", "fileSize": "123 bytes"}</script>
    <a href="https://github.com/jjang-ai/mlxstudio/releases/download/v9.8.6/vMLX-9.8.6-tahoe-arm64.dmg">9.8.6</a>
    <a href="https://github.com/jjang-ai/mlxstudio/releases/download/v9.8.6/vMLX-9.8.6-sequoia-arm64.dmg">compat</a>
    """
    (site / "download/index.html").write_text(old_html)
    updater = {
        "version": "9.8.7",
        "downloads": {
            "tahoe": {
                "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v9.8.7/vMLX-9.8.7-tahoe-arm64.dmg",
                "bytes": 999,
            },
            "sequoia": {
                "url": "https://github.com/jjang-ai/mlxstudio/releases/download/v9.8.7/vMLX-9.8.7-sequoia-arm64.dmg",
                "bytes": 888,
            },
        },
    }
    updater_path = tmp_path / "latest.json"
    updater_path.write_text(json.dumps(updater))
    backups = tmp_path / "backups"

    subprocess.run(
        [
            sys.executable,
            str(SITE_SCRIPT),
            "--site-root",
            str(site),
            "--updater",
            str(updater_path),
            "--backup-root",
            str(backups),
        ],
        check=True,
    )

    html = (site / "download/index.html").read_text()
    assert "9.8.6" not in html
    assert "vMLX-9.8.7-tahoe-arm64.dmg" in html
    assert "vMLX-9.8.7-sequoia-arm64.dmg" in html
    assert '"fileSize": "999 bytes"' in html
    assert (backups / "vmlx-9.8.6-before-9.8.7/latest.json").is_file()
    assert json.loads((site / "update/latest.json").read_text()) == updater

    # A resumed workflow must verify-and-skip the exact deployed version rather
    # than fail because its versioned backup already exists.
    subprocess.run(
        [
            sys.executable,
            str(SITE_SCRIPT),
            "--site-root",
            str(site),
            "--updater",
            str(updater_path),
            "--backup-root",
            str(backups),
        ],
        check=True,
    )
    assert len(list(backups.iterdir())) == 1


def test_workflows_are_manual_pinned_and_keep_secret_boundaries():
    workflow_dir = ROOT / ".github/workflows"
    files = {
        path.name: (path.read_text(), yaml.load(path.read_text(), Loader=yaml.BaseLoader))
        for path in workflow_dir.glob("*.yml")
    }
    assert set(files) == {
        "dev-build.yml",
        "release-candidate.yml",
        "publish-release.yml",
        "release-automation-ci.yml",
    }
    for name, (source, parsed) in files.items():
        if name == "release-automation-ci.yml":
            assert set(parsed["on"]) == {"push", "pull_request"}
            assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in source
            continue
        assert set(parsed["on"]) == {"workflow_dispatch"}, name
        assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in source
        assert "PYPI_API_TOKEN" not in source
        assert "secrets.PYPI" not in source
        action_uses = re.findall(r"^\s*uses:\s*([^\s]+)\s*$", source, re.MULTILINE)
        assert action_uses
        assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in action_uses)

    dev = files["dev-build.yml"][0]
    assert "VMLX_DEV_UNSIGNED: \"1\"" in dev
    assert "codex_ui_only" in dev
    assert "notarize-release-dmgs.sh" not in dev

    candidate = files["release-candidate.yml"][0]
    assert "environment: apple-release-signing" in candidate
    assert "VMLX_RELEASE_SCOPE: r20_production" in candidate
    assert "build-release-dmgs.sh all" in candidate
    assert "notarize-release-dmgs.sh" in candidate
    assert 'export VMLX_R20_EXPECTED_SOURCE_COMMIT="$SOURCE_SHA"' in candidate
    assert 'export VMLX_R20_EXPECTED_SOURCE_TREE="$SOURCE_TREE"' in candidate
    assert 'export VMLX_R20_EXPECTED_PREFLIGHT_SHA256="$preflight_sha"' in candidate
    assert "jjang-ai/jangq" in candidate
    assert "private_evidence_root" in candidate
    assert "if: ${{ !inputs.scoped_prepackage_receipt }}" in candidate
    assert "VMLX_SCOPED_PREPACKAGE_RECEIPT: ${{ inputs.scoped_prepackage_receipt }}" in candidate
    assert '[[ "$SCOPED_PREPACKAGE_RECEIPT" == "$PRIVATE_EVIDENCE_ROOT/"* ]]' in candidate
    assert "private_evidence_root }}" not in candidate.split("Upload immutable release candidate", 1)[1]
    assert "vars.APPLE_NOTARY_KEYCHAIN_PROFILE" in candidate
    assert "vars.APPLE_USE_EXISTING_SIGNING_IDENTITY" in candidate
    assert "steps.apple.outputs.notary_keychain" in candidate
    assert "import tomllib" not in candidate
    assert "pathlib.Path('pyproject.toml').read_text()" in candidate
    assert 'UV_BIN="$(command -v uv)"' in candidate
    assert '"$UV_BIN" venv --seed --python 3.13 .venv' in candidate
    assert 'Path(sys.executable).absolute()' in candidate
    assert "RELEASE_PYTHON=/opt/homebrew/bin/python3" not in candidate
    assert "/usr/bin/python3 -m venv" not in candidate

    signing = (ROOT / ".github/scripts/setup_apple_signing.sh").read_text()
    assert "APPLE_NOTARY_EXISTING_PROFILE" in signing
    assert "APPLE_USE_EXISTING_SIGNING_IDENTITY" in signing
    assert "Developer ID Application: ShieldStack LLC (55KGF2S5AY)" in signing
    assert 'notary_keychain=""' in signing
    assert 'echo "notary_keychain=$notary_keychain"' in signing
    assert 'notarytool history' in signing

    publish = files["publish-release.yml"][0]
    assert "jjang-ai/vmlx" in publish and "jjang-ai/mlxstudio" in publish
    assert "environment: production-release" in publish
    assert "environment: pypi" in publish
    assert "id-token: write" in publish
    assert "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in publish
    assert "Publish without a stored PyPI token" in publish
    assert "always() && inputs.deploy_website && needs.update-manifests.result == 'success'" in publish
    assert "publish-pypi.yml -R jjang-ai/jangq" in publish
    assert "cmp repo-vmlx/latest.json repo-mlxstudio/latest.json" in publish
    assert "TRUSTED_PUBLISHER_CONFIGURED" in publish
    assert "secrets.RELEASE_PAT" in publish
    assert "release_metadata.py check-pypi" in publish
    assert "publish_required == 'true'" in publish
    assert "Mozilla/5.0 vMLX-Release-Verification" in publish
    assert "Check out immutable release helper" in publish


def test_pypi_recovery_uses_public_inputs_without_a_candidate_run():
    import yaml
    workflow = yaml.load((ROOT / ".github/workflows/publish-release.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["workflow_dispatch"]["inputs"]["candidate_run_id"]["required"] == "false"
    steps = workflow["jobs"]["update-manifests"]["steps"]
    download = next(step for step in steps if step["name"] == "Download exact candidate artifact")
    assert download["if"] == "${{ !inputs.pypi_recovery }}"
    recovery = next(step for step in steps if step["name"] == "Recover verified public updater inputs")
    assert recovery["if"] == "${{ inputs.pypi_recovery }}"
    assert 'metadata["source"]["commit"] == os.environ["SOURCE_SHA"]' in recovery["run"]
    assert 'asset["digest"] == "sha256:" + digest' in recovery["run"]
    assert 'release["draft"] is False' in recovery["run"]
    assert 'release["prerelease"] is False' in recovery["run"]
    assert 'candidate/release-notes.md' in recovery["run"]
    validation = workflow["jobs"]["validate"]["steps"]
    guard = next(step for step in validation if step["name"] == "Require a candidate run for normal promotion")
    assert '^[1-9][0-9]*$' in guard["run"]


@pytest.mark.parametrize("mismatch", [None, "source", "jang", "tag", "draft", "digest", "size", "duplicate", "notes"])
def test_public_recovery_checks_production_payload(tmp_path, monkeypatch, mismatch):
    import hashlib
    workflow = yaml.load((ROOT / ".github/workflows/publish-release.yml").read_text(), Loader=yaml.BaseLoader)
    step = next(s for s in workflow["jobs"]["update-manifests"]["steps"]
                if s["name"] == "Recover verified public updater inputs")
    code = step["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    monkeypatch.chdir(tmp_path)
    for name, value in {"VERSION": "9.8.6", "SOURCE_SHA": "a" * 40, "JANG_SHA": "b" * 40}.items():
        monkeypatch.setenv(name, value)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    metadata = {"version": "9.8.6", "source": {"commit": "a" * 40}, "jangq": {"commit": "b" * 40},
                "artifacts": {}, "python_distributions": {}}
    assets = []
    for flavor in ("sequoia", "tahoe"):
        row = {"filename": flavor + ".dmg", "bytes": 11, "sha256": "c" * 64,
               "blockmap_filename": flavor + ".blockmap", "blockmap_bytes": 3, "blockmap_sha256": "d" * 64}
        metadata["artifacts"][flavor] = row
        assets.extend([{"name": row["filename"], "size": 11, "digest": "sha256:" + "c" * 64},
                       {"name": row["blockmap_filename"], "size": 3, "digest": "sha256:" + "d" * 64}])
    for kind in ("wheel", "sdist"):
        metadata["python_distributions"][kind] = {"filename": kind, "bytes": 7, "sha256": "e" * 64}
        assets.append({"name": kind, "size": 7, "digest": "sha256:" + "e" * 64})
    if mismatch in ("source", "jang"):
        metadata["source" if mismatch == "source" else "jangq"]["commit"] = "f" * 40
    raw = json.dumps(metadata).encode()
    (candidate / "release-info.json").write_bytes(raw)
    assets.append({"name": "release-info.json", "size": len(raw), "digest": "sha256:" + hashlib.sha256(raw).hexdigest()})
    release = {"tag_name": "v9.8.6", "draft": False, "prerelease": False,
               "assets": assets, "body": "# vMLX 9.8.6\n\nExact release notes.\n"}
    if mismatch == "tag": release["tag_name"] = "v9.8.5"
    if mismatch == "draft": release["draft"] = True
    if mismatch == "digest": assets[0]["digest"] = "sha256:" + "f" * 64
    if mismatch == "size": assets[0]["size"] += 1
    if mismatch == "duplicate": assets.append(dict(assets[0]))
    if mismatch == "notes": release["body"] = "# vMLX 9.8.5\n"
    (candidate / "github-release.json").write_text(json.dumps(release))
    if mismatch:
        with pytest.raises(AssertionError): exec(compile(code, "workflow-recovery", "exec"), {})
        assert not (candidate / "release-notes.md").exists()
    else:
        exec(compile(code, "workflow-recovery", "exec"), {})
        assert (candidate / "release-notes.md").read_text() == release["body"]


def test_after_pack_detaches_engine_source_hardlinks(tmp_path):
    source = tmp_path / "source.py"
    packaged = tmp_path / "app" / "vmlx-engine-source" / "vmlx_engine" / "source.py"
    packaged.parent.mkdir(parents=True)
    source.write_bytes(b"release-source-bytes\n")
    os.link(source, packaged)
    assert source.stat().st_nlink == 2

    script = ROOT / "panel/scripts/electron-builder-after-pack.cjs"
    subprocess.run(
        [
            "/opt/homebrew/bin/node",
            "-e",
            (
                "const hook=require(process.argv[1]);"
                "const rows=hook.detachHardlinkedTree(process.argv[2]);"
                "if(rows.length!==1) process.exit(9);"
            ),
            str(script),
            str(tmp_path / "app" / "vmlx-engine-source"),
        ],
        check=True,
    )

    assert source.stat().st_nlink == 1
    assert packaged.stat().st_nlink == 1
    assert source.read_bytes() == packaged.read_bytes() == b"release-source-bytes\n"
    packaged.write_bytes(b"packaged-copy-only\n")
    assert source.read_bytes() == b"release-source-bytes\n"



def test_unsigned_dev_escape_hatch_is_impossible_in_production():
    source = (ROOT / "panel/scripts/build-release-dmgs.sh").read_text()
    assert 'DEV_UNSIGNED="${VMLX_DEV_UNSIGNED:-0}"' in source
    assert '"$DEV_UNSIGNED" == "1" && "$RELEASE_SCOPE" != "codex_ui_only"' in source
    assert "unsigned DMGs are allowed only" in source
    assert "--config.mac.identity=null" in source


def test_release_automation_excludes_private_docs_and_machine_paths():
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "docs/internal/ISSUE-LEDGER.md"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ignored.returncode == 0
    assert ignored.stdout.strip() == "docs/internal/ISSUE-LEDGER.md"

    tracked = subprocess.run(
        ["git", "ls-files", "docs/internal"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert tracked.stdout == ""

    public_release_files = [
        *(ROOT / ".github/workflows").glob("*.yml"),
        *(ROOT / ".github/scripts").glob("*"),
        ROOT / "docs/development/release-automation.md",
    ]
    public_text = "\n".join(
        path.read_text(errors="replace")
        for path in public_release_files
        if path.is_file()
    )
    forbidden = (
        "/Users/",
        "/private/tmp",
        "docs/internal",
        "ISSUE-LEDGER",
        "TO-DO-AFTER-RELEASE",
        "bench-keep",
        "remote-cdp",
        ".ndjson",
        "screenshot",
    )
    assert not [needle for needle in forbidden if needle in public_text]

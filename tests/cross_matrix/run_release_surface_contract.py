#!/usr/bin/env python3
"""Validate release/updater surface state without publishing anything.

This preflight is intentionally local-only by default. It prevents the release
prep branch from silently advancing `latest.json` beyond the source version, or
from bumping it to the source version without a matching URL, checksum, and
release-note version. Live public PyPI/GitHub/update-feed checks are handled by
the human/operator step before publishing because they can change at any time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Any, Callable


DEFAULT_OUT = Path(
    "build/current-release-surface-contract-20260602-v154-live-public-after-site-fix.json"
)

SOURCE_HASH_FILES = (
    "latest.json",
    "panel/package.json",
    "panel/src/main/update-checker.ts",
    "pyproject.toml",
    ".github/workflows/publish-pypi.yml",
    "tests/cross_matrix/run_release_surface_contract.py",
    "tests/test_release_surface_contract.py",
)

FetchJson = Callable[[str], dict[str, Any]]
FetchHeaders = Callable[[str], dict[str, str]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_version(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(value or ""))
    return tuple(int(part) for part in parts[:4])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "vmlx-release-surface-contract",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_headers(url: str) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "User-Agent": "vmlx-release-surface-contract",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return {key.lower(): value for key, value in response.headers.items()}


def _current_git_head(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _source_versions(root: Path) -> dict[str, str | None]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    panel_pkg = _load_json(root / "panel/package.json")
    return {
        "pyproject": pyproject.get("project", {}).get("version") or pyproject.get("version"),
        "panel_package": panel_pkg.get("version"),
    }


def _local_updater_checks(source_version: str, latest: dict[str, Any]) -> dict[str, bool]:
    latest_version = str(latest.get("version") or "")
    latest_url = str(latest.get("url") or "")
    latest_sha = str(latest.get("sha256") or "")
    latest_notes = str(latest.get("notes") or "")
    source_tuple = _parse_version(source_version)
    latest_tuple = _parse_version(latest_version)
    bumped_to_source = latest_version == source_version
    complete_bumped_release = (
        bumped_to_source
        and all(
            isinstance(latest.get(key), str) and latest.get(key)
            for key in ("version", "url", "sha256", "notes")
        )
        and f"v{source_version}" in latest_url
        and re.fullmatch(r"[0-9a-fA-F]{64}", latest_sha) is not None
        and source_version in latest_notes
    )
    downloads = latest.get("downloads")
    sequoia = downloads.get("sequoia") if isinstance(downloads, dict) else None
    tahoe = downloads.get("tahoe") if isinstance(downloads, dict) else None
    platform_downloads_valid = True
    if bumped_to_source:
        platform_downloads_valid = all(
            isinstance(row, dict)
            and isinstance(row.get("url"), str)
            and f"v{source_version}" in row.get("url", "")
            and flavor in row.get("url", "")
            and isinstance(row.get("sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", row.get("sha256", "")) is not None
            for flavor, row in (("sequoia", sequoia), ("tahoe", tahoe))
        )
    return {
        "local_updater_not_ahead_of_source": bool(latest_tuple <= source_tuple),
        "staged_source_version_not_public": bool(latest_tuple < source_tuple),
        "local_updater_release_state_valid": bool(
            latest_tuple < source_tuple or (complete_bumped_release and platform_downloads_valid)
        ),
        "local_updater_platform_downloads_valid": bool(platform_downloads_valid),
        "local_updater_has_required_fields": all(
            isinstance(latest.get(key), str) and latest.get(key)
            for key in ("version", "url", "sha256", "notes")
        ),
        "local_updater_url_matches_version": (
            not bumped_to_source or f"v{source_version}" in latest_url
        ),
        "local_updater_sha256_valid": (
            not bumped_to_source or re.fullmatch(r"[0-9a-fA-F]{64}", latest_sha) is not None
        ),
        "local_updater_notes_match_version": (
            not bumped_to_source or source_version in latest_notes
        ),
    }


def _updater_matches_local(public: dict[str, Any], local: dict[str, Any]) -> bool:
    return all(
        str(public.get(key) or "") == str(local.get(key) or "")
        for key in ("version", "url", "sha256")
    )


def _public_release_checks(
    source_version: str,
    latest: dict[str, Any],
    fetch_json: FetchJson,
    fetch_headers: FetchHeaders,
    current_revision: str | None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    raw_latest_url = "https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json"
    site_latest_url = "https://mlx.studio/update/latest.json"
    pypi_url = f"https://pypi.org/pypi/vmlx/{source_version}/json"
    github_release_url = (
        f"https://api.github.com/repos/jjang-ai/mlxstudio/releases/tags/v{source_version}"
    )
    github_source_tag_ref_url = (
        f"https://api.github.com/repos/jjang-ai/vmlx/git/ref/tags/v{source_version}"
    )
    github_source_tag_url_prefix = "https://api.github.com/repos/jjang-ai/vmlx/git/tags/"

    public: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    try:
        raw_latest = fetch_json(raw_latest_url)
        public["raw_latest"] = {
            "version": raw_latest.get("version"),
            "url": raw_latest.get("url"),
            "sha256": raw_latest.get("sha256"),
        }
        checks["public_raw_updater_matches_local"] = _updater_matches_local(
            raw_latest, latest
        )
    except Exception as exc:  # pragma: no cover - exercised by live failures.
        public["raw_latest_error"] = repr(exc)
        checks["public_raw_updater_matches_local"] = False

    try:
        site_latest = fetch_json(site_latest_url)
        public["site_latest"] = {
            "version": site_latest.get("version"),
            "url": site_latest.get("url"),
            "sha256": site_latest.get("sha256"),
        }
        checks["public_site_updater_matches_local"] = _updater_matches_local(
            site_latest, latest
        )
    except Exception as exc:  # pragma: no cover - exercised by live failures.
        public["site_latest_error"] = repr(exc)
        checks["public_site_updater_matches_local"] = False

    try:
        site_headers = fetch_headers(site_latest_url)
        cache_control = site_headers.get("cache-control", "")
        pragma = site_headers.get("pragma", "")
        expires = site_headers.get("expires", "")
        public["site_latest_headers"] = {
            "cache-control": cache_control,
            "pragma": pragma,
            "expires": expires,
            "cf-cache-status": site_headers.get("cf-cache-status"),
            "access-control-allow-origin": site_headers.get("access-control-allow-origin"),
        }
        cache_directives = {
            item.strip().lower()
            for item in cache_control.split(",")
            if item.strip()
        }
        checks["public_site_updater_cache_headers_safe"] = (
            "no-cache" in cache_directives
            and "no-store" in cache_directives
            and "must-revalidate" in cache_directives
            and pragma.lower() == "no-cache"
            and expires == "0"
        )
    except Exception as exc:  # pragma: no cover - exercised by live failures.
        public["site_latest_headers_error"] = repr(exc)
        checks["public_site_updater_cache_headers_safe"] = False

    expected_wheel = f"vmlx-{source_version}-py3-none-any.whl"
    expected_sdist = f"vmlx-{source_version}.tar.gz"
    try:
        pypi = fetch_json(pypi_url)
        pypi_files = sorted(str(file.get("filename") or "") for file in pypi.get("urls", []))
        public["pypi"] = {
            "version": pypi.get("info", {}).get("version"),
            "files": pypi_files,
        }
        checks["public_pypi_has_release_files"] = (
            pypi.get("info", {}).get("version") == source_version
            and expected_wheel in pypi_files
            and expected_sdist in pypi_files
        )
    except Exception as exc:  # pragma: no cover - exercised by live failures.
        public["pypi_error"] = repr(exc)
        checks["public_pypi_has_release_files"] = False

    updater_asset_name = str(latest.get("url") or "").rsplit("/", 1)[-1]
    expected_digest = "sha256:" + str(latest.get("sha256") or "")
    expected_assets = {
        updater_asset_name: expected_digest,
    }
    downloads = latest.get("downloads")
    if isinstance(downloads, dict):
        for row in downloads.values():
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "")
            sha = str(row.get("sha256") or "")
            name = url.rsplit("/", 1)[-1]
            if name:
                expected_assets[name] = "sha256:" + sha if sha else ""
    try:
        release = fetch_json(github_release_url)
        assets = release.get("assets", [])
        asset_summaries = [
            {
                "name": asset.get("name"),
                "digest": asset.get("digest"),
                "state": asset.get("state"),
            }
            for asset in assets
        ]
        public["github_release"] = {
            "draft": release.get("draft"),
            "prerelease": release.get("prerelease"),
            "assets": asset_summaries,
        }
        checks["public_github_release_published"] = (
            release.get("draft") is False and release.get("prerelease") is False
        )
        checks["public_github_release_has_updater_asset"] = any(
            asset.get("name") == updater_asset_name
            and (
                not asset.get("digest")
                or str(asset.get("digest")) == expected_digest
            )
            for asset in assets
        )
        checks["public_github_release_has_all_manifest_download_assets"] = all(
            any(
                asset.get("name") == name
                and (
                    not expected
                    or not asset.get("digest")
                    or str(asset.get("digest")) == expected
                )
                for asset in assets
            )
            for name, expected in expected_assets.items()
        )
    except Exception as exc:  # pragma: no cover - exercised by live failures.
        public["github_release_error"] = repr(exc)
        checks["public_github_release_published"] = False
        checks["public_github_release_has_updater_asset"] = False
        checks["public_github_release_has_all_manifest_download_assets"] = False

    try:
        tag_ref = fetch_json(github_source_tag_ref_url)
        tag_object = tag_ref.get("object") if isinstance(tag_ref.get("object"), dict) else {}
        tag_sha = str(tag_object.get("sha") or "")
        tag_type = tag_object.get("type")
        resolved_sha = tag_sha
        resolved_type = tag_type
        if tag_sha and tag_type == "tag":
            annotated_tag = fetch_json(github_source_tag_url_prefix + tag_sha)
            annotated_object = (
                annotated_tag.get("object")
                if isinstance(annotated_tag.get("object"), dict)
                else {}
            )
            resolved_sha = str(annotated_object.get("sha") or "")
            resolved_type = annotated_object.get("type")
        public["github_source_release_tag"] = {
            "sha": tag_sha,
            "type": tag_type,
            "resolved_sha": resolved_sha,
            "resolved_type": resolved_type,
            "current_revision": current_revision,
        }
        checks["public_github_source_release_tag_matches_source_head"] = bool(
            current_revision and resolved_sha == current_revision
        )
    except Exception as exc:  # pragma: no cover - exercised by live failures.
        public["github_source_release_tag_error"] = repr(exc)
        checks["public_github_source_release_tag_matches_source_head"] = False

    return checks, public


def build_artifact(
    root: Path,
    *,
    live_public: bool = False,
    fetch_json: FetchJson = _fetch_json,
    fetch_headers: FetchHeaders = _fetch_headers,
    current_revision: str | None = None,
) -> dict[str, Any]:
    versions = _source_versions(root)
    source_version = versions["pyproject"]
    panel_version = versions["panel_package"]
    latest = _load_json(root / "latest.json")

    checks = {
        "source_version_consistent": bool(source_version and source_version == panel_version),
        **_local_updater_checks(str(source_version or ""), latest),
    }
    public_surfaces: dict[str, Any] = {}
    current_revision = current_revision or _current_git_head(root)
    if live_public and source_version:
        public_checks, public_surfaces = _public_release_checks(
            str(source_version), latest, fetch_json, fetch_headers, current_revision
        )
        checks.update(public_checks)
    status_check_names = {
        "source_version_consistent",
        "local_updater_not_ahead_of_source",
        "local_updater_release_state_valid",
        "local_updater_has_required_fields",
        "local_updater_url_matches_version",
        "local_updater_sha256_valid",
        "local_updater_notes_match_version",
        "local_updater_platform_downloads_valid",
    }
    if live_public:
        status_check_names.update(
            {
                "public_raw_updater_matches_local",
                "public_site_updater_matches_local",
                "public_site_updater_cache_headers_safe",
                "public_pypi_has_release_files",
                "public_github_release_published",
                "public_github_release_has_updater_asset",
                "public_github_release_has_all_manifest_download_assets",
                "public_github_source_release_tag_matches_source_head",
            }
        )
    status_failed_checks = sorted(
        name for name in status_check_names if not checks.get(name, False)
    )
    informational_false_checks = sorted(
        name for name, passed in checks.items()
        if not passed and name not in status_check_names
    )
    status = "pass" if not status_failed_checks else "fail"
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "checks": checks,
        "status_failed_checks": status_failed_checks,
        "informational_false_checks": informational_false_checks,
        "source_versions": versions,
        "local_latest": {
            "version": latest.get("version"),
            "url": latest.get("url"),
            "sha256": latest.get("sha256"),
            "downloads": latest.get("downloads"),
            "notes_head": str(latest.get("notes") or "").splitlines()[:8],
        },
        "live_public": live_public,
        "public_surfaces": public_surfaces,
        "current_revision": current_revision,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--live-public",
        action="store_true",
        help="Also verify public GitHub raw/latest, mlx.studio latest, PyPI, and GitHub release asset state.",
    )
    args = parser.parse_args()

    artifact = build_artifact(args.root, live_public=args.live_public)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("checks=" + json.dumps(artifact["checks"], sort_keys=True))
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

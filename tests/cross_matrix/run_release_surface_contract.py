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
import time
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-release-surface-contract-20260521.json")

SOURCE_HASH_FILES = (
    "latest.json",
    "panel/package.json",
    "panel/src/main/update-checker.ts",
    "pyproject.toml",
    "tests/cross_matrix/run_release_surface_contract.py",
    "tests/test_release_surface_contract.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_version(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(value or ""))
    return tuple(int(part) for part in parts[:4])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return {
        "local_updater_not_ahead_of_source": bool(latest_tuple <= source_tuple),
        "staged_source_version_not_public": bool(latest_tuple < source_tuple),
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


def build_artifact(root: Path) -> dict[str, Any]:
    versions = _source_versions(root)
    source_version = versions["pyproject"]
    panel_version = versions["panel_package"]
    latest = _load_json(root / "latest.json")

    checks = {
        "source_version_consistent": bool(source_version and source_version == panel_version),
        **_local_updater_checks(str(source_version or ""), latest),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "checks": checks,
        "source_versions": versions,
        "local_latest": {
            "version": latest.get("version"),
            "url": latest.get("url"),
            "sha256": latest.get("sha256"),
            "notes_head": str(latest.get("notes") or "").splitlines()[:8],
        },
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
    args = parser.parse_args()

    artifact = build_artifact(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("checks=" + json.dumps(artifact["checks"], sort_keys=True))
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

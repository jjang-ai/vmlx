#!/usr/bin/env python3
"""Create and verify public vMLX release metadata.

This helper intentionally emits only public provenance. Private proof paths,
Apple credentials, notary payloads, and local machine paths must never enter
the candidate artifact.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
SHA_RE = re.compile(r"[0-9a-f]{40}")
FLAVORS = ("sequoia", "tahoe")
PUBLIC_REPOSITORY = "jjang-ai/vmlx"
PUBLIC_JANG_REPOSITORY = "jjang-ai/jangq"
PUBLIC_BINARY_REPOSITORY = "jjang-ai/mlxstudio"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def require_version(value: str) -> str:
    if not VERSION_RE.fullmatch(value):
        fail(f"invalid release version: {value!r}")
    return value


def require_sha(value: str, label: str) -> str:
    value = value.lower()
    if not SHA_RE.fullmatch(value):
        fail(f"{label} must be a full lowercase Git SHA, got {value!r}")
    return value


def hash_file(path: Path) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            sha256.update(chunk)
            sha512.update(chunk)
    return sha256.hexdigest(), base64.b64encode(sha512.digest()).decode("ascii"), size


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def artifact_record(dist: Path, version: str, flavor: str) -> dict[str, Any]:
    filename = f"vMLX-{version}-{flavor}-arm64.dmg"
    blockmap_filename = f"{filename}.blockmap"
    dmg = dist / filename
    blockmap = dist / blockmap_filename
    if not dmg.is_file() or not blockmap.is_file():
        fail(f"missing {flavor} DMG or blockmap in {dist}")
    dmg_sha256, dmg_sha512, dmg_size = hash_file(dmg)
    blockmap_sha256, _, blockmap_size = hash_file(blockmap)
    runtime = load_json(Path(__file__).resolve().parents[2] / "panel/scripts/mlx-runtime.json")
    platform = "macosx_26_0_arm64" if flavor == "tahoe" else "macosx_14_0_arm64"
    minimum = runtime["platforms"][platform]["minimum_system_version"]
    return {
        "minimumSystemVersion": minimum,
        "filename": filename,
        "bytes": dmg_size,
        "sha256": dmg_sha256,
        "sha512": dmg_sha512,
        "blockmap_filename": blockmap_filename,
        "blockmap_bytes": blockmap_size,
        "blockmap_sha256": blockmap_sha256,
    }


def python_distribution_records(dist: Path, version: str) -> dict[str, Any]:
    expected = {
        "wheel": f"vmlx-{version}-py3-none-any.whl",
        "sdist": f"vmlx-{version}.tar.gz",
    }
    actual = {path.name for path in dist.iterdir() if path.is_file()}
    if actual != set(expected.values()):
        fail(
            "Python distribution set is not exact: "
            f"expected {sorted(expected.values())}, got {sorted(actual)}"
        )
    records: dict[str, Any] = {}
    for kind, filename in expected.items():
        sha256, _, size = hash_file(dist / filename)
        records[kind] = {"filename": filename, "bytes": size, "sha256": sha256}
    return records


def read_jang_version(jang_tools: Path) -> str:
    import tomllib

    pyproject = jang_tools / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            value = tomllib.load(handle)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        fail(f"cannot read JANG version from {pyproject}: {exc}")
    return require_version(str(value))


def git_output(root: Path, *args: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 and not allow_failure:
        fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def create_metadata(args: argparse.Namespace) -> None:
    version = require_version(args.version)
    source_sha = require_sha(args.source_sha, "source SHA")
    source_tree = require_sha(args.source_tree, "source tree")
    jang_sha = require_sha(args.jang_sha, "JANG SHA")
    final_manifest_sha256 = args.final_notary_manifest_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", final_manifest_sha256):
        fail("final notary manifest SHA256 must be 64 lowercase hex characters")

    dist = args.dist.resolve()
    records = {flavor: artifact_record(dist, version, flavor) for flavor in FLAVORS}
    python_records = python_distribution_records(args.python_dist.resolve(), version)
    expected_names = {
        record[key]
        for record in records.values()
        for key in ("filename", "blockmap_filename")
    }
    actual_names = {
        path.name
        for path in dist.iterdir()
        if path.is_file() and (path.name.endswith(".dmg") or path.name.endswith(".dmg.blockmap"))
    }
    if actual_names != expected_names:
        fail(f"candidate DMG set is not exact: expected {sorted(expected_names)}, got {sorted(actual_names)}")

    jang_version = read_jang_version(args.jang_tools)
    metadata = {
        "schema_version": 1,
        "version": version,
        "tag": f"v{version}",
        "source": {
            "repository": PUBLIC_REPOSITORY,
            "commit": source_sha,
            "tree": source_tree,
        },
        "jangq": {
            "repository": PUBLIC_JANG_REPOSITORY,
            "commit": jang_sha,
            "version": jang_version,
        },
        "notarization": {
            "status": "accepted_and_stapled",
            "private_manifest_sha256": final_manifest_sha256,
        },
        "artifacts": records,
        "python_distributions": python_records,
        "updater": {"default": "tahoe", "compatibility": "sequoia"},
    }
    write_json(args.output, metadata)

    previous_tag = args.previous_tag
    if not previous_tag:
        previous_tag = git_output(
            args.root,
            "describe",
            "--tags",
            "--abbrev=0",
            f"{source_sha}^",
            allow_failure=True,
        )
    revision_range = f"{previous_tag}..{source_sha}" if previous_tag else source_sha
    changes = git_output(
        args.root,
        "log",
        "--no-merges",
        "--reverse",
        "--format=- %s (`%h`)",
        revision_range,
    )
    if not changes:
        changes = "- No source commits were found in the selected release range."

    notes = [
        f"# vMLX {version}",
        "",
        "## Changes",
        "",
        changes,
        "",
        "## Runtime provenance",
        "",
        f"- vMLX source: `{source_sha}`",
        f"- JANG runtime: `{jang_sha}` (`jang {jang_version}`)",
        "- Tahoe/macOS 26 is the default build; Sequoia is the compatibility build.",
        "- Both DMGs are Developer ID signed, independently notarized, stapled, and Gatekeeper checked by the candidate workflow.",
        "",
        "## Downloads",
        "",
    ]
    for flavor in ("tahoe", "sequoia"):
        record = records[flavor]
        notes.extend(
            [
                f"- **{flavor.title()}**: `{record['filename']}`",
                f"  - SHA-256: `{record['sha256']}`",
                f"  - Size: `{record['bytes']}` bytes",
            ]
        )
    notes.extend(
        [
            "",
            "## Python distributions",
            "",
            f"- Wheel: `{python_records['wheel']['filename']}` — SHA-256 `{python_records['wheel']['sha256']}`",
            f"- Source: `{python_records['sdist']['filename']}` — SHA-256 `{python_records['sdist']['sha256']}`",
        ]
    )
    args.notes_output.parent.mkdir(parents=True, exist_ok=True)
    args.notes_output.write_text("\n".join(notes) + "\n", encoding="utf-8")


def verify_metadata(args: argparse.Namespace) -> None:
    metadata = load_json(args.metadata)
    expected_version = require_version(args.version)
    expected_source = require_sha(args.source_sha, "source SHA")
    expected_jang = require_sha(args.jang_sha, "JANG SHA")
    checks = {
        "schema_version": metadata.get("schema_version") == 1,
        "version": metadata.get("version") == expected_version,
        "tag": metadata.get("tag") == f"v{expected_version}",
        "source_repository": metadata.get("source", {}).get("repository") == PUBLIC_REPOSITORY,
        "source_commit": metadata.get("source", {}).get("commit") == expected_source,
        "jang_repository": metadata.get("jangq", {}).get("repository") == PUBLIC_JANG_REPOSITORY,
        "jang_commit": metadata.get("jangq", {}).get("commit") == expected_jang,
        "notarized": metadata.get("notarization", {}).get("status") == "accepted_and_stapled",
        "tahoe_default": metadata.get("updater", {}).get("default") == "tahoe",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        fail(f"release metadata contract failed: {', '.join(failed)}")

    dist = args.dist.resolve()
    for flavor in FLAVORS:
        expected = artifact_record(dist, expected_version, flavor)
        if metadata.get("artifacts", {}).get(flavor) != expected:
            fail(f"{flavor} artifact bytes do not match release metadata")
    expected_python = python_distribution_records(args.python_dist.resolve(), expected_version)
    if metadata.get("python_distributions") != expected_python:
        fail("Python distribution bytes do not match release metadata")

    encoded = json.dumps(metadata, sort_keys=True).lower()
    forbidden = ("private_evidence", "runner_temp", "keychain", "notary-result", "/users/")
    leaked = [needle for needle in forbidden if needle in encoded]
    if leaked:
        fail(f"public metadata contains private-path vocabulary: {', '.join(leaked)}")


def write_updater(args: argparse.Namespace) -> None:
    metadata = load_json(args.metadata)
    version = require_version(str(metadata.get("version", "")))
    artifacts = metadata.get("artifacts", {})
    notes = args.notes.read_text(encoding="utf-8").strip()
    if not notes.startswith(f"# vMLX {version}"):
        fail("release notes do not match metadata version")

    downloads: dict[str, Any] = {}
    for flavor in FLAVORS:
        record = artifacts.get(flavor, {})
        filename = record.get("filename")
        if filename != f"vMLX-{version}-{flavor}-arm64.dmg":
            fail(f"invalid {flavor} filename in metadata")
        downloads[flavor] = {
            "minimumSystemVersion": record["minimumSystemVersion"],
            "url": f"https://github.com/{PUBLIC_BINARY_REPOSITORY}/releases/download/v{version}/{filename}",
            "bytes": record.get("bytes"),
            "sha256": record.get("sha256"),
            "sha512": record.get("sha512"),
        }
    updater = {
        "version": version,
        "url": downloads["tahoe"]["url"],
        "sha256": downloads["tahoe"]["sha256"],
        "sha512": downloads["tahoe"]["sha512"],
        "downloads": downloads,
        "notes": notes,
    }
    write_json(args.output, updater)


def classify_pypi_release(
    metadata: dict[str, Any], payload: dict[str, Any] | None
) -> str:
    """Return missing/exact and reject a conflicting immutable PyPI version."""

    expected = {
        record["filename"]: (record["bytes"], record["sha256"])
        for record in metadata.get("python_distributions", {}).values()
    }
    if len(expected) != 2:
        fail("release metadata must contain exactly one wheel and one source archive")
    if payload is None:
        return "missing"

    actual = {
        row.get("filename"): (
            row.get("size"),
            row.get("digests", {}).get("sha256"),
        )
        for row in payload.get("urls", [])
    }
    if actual != expected:
        fail(
            "PyPI version exists with a different artifact set: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    return "exact"


def check_pypi(args: argparse.Namespace) -> None:
    metadata = load_json(args.metadata)
    version = require_version(args.version)
    if metadata.get("version") != version:
        fail("PyPI check version does not match release metadata")

    url = f"https://pypi.org/pypi/vmlx/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            fail(f"PyPI release query failed with HTTP {exc.code}")
        payload = None
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"PyPI release query failed: {exc}")

    state = classify_pypi_release(metadata, payload)
    publish_required = "true" if state == "missing" else "false"
    output = args.github_output
    if output is not None:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(f"state={state}\n")
            handle.write(f"publish_required={publish_required}\n")
    print(f"PyPI vmlx {version}: {state}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--dist", type=Path, required=True)
    create.add_argument("--python-dist", type=Path, required=True)
    create.add_argument("--version", required=True)
    create.add_argument("--source-sha", required=True)
    create.add_argument("--source-tree", required=True)
    create.add_argument("--jang-sha", required=True)
    create.add_argument("--jang-tools", type=Path, required=True)
    create.add_argument("--final-notary-manifest-sha256", required=True)
    create.add_argument("--previous-tag")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--notes-output", type=Path, required=True)
    create.set_defaults(func=create_metadata)

    verify = sub.add_parser("verify")
    verify.add_argument("--metadata", type=Path, required=True)
    verify.add_argument("--dist", type=Path, required=True)
    verify.add_argument("--python-dist", type=Path, required=True)
    verify.add_argument("--version", required=True)
    verify.add_argument("--source-sha", required=True)
    verify.add_argument("--jang-sha", required=True)
    verify.set_defaults(func=verify_metadata)

    updater = sub.add_parser("write-updater")
    updater.add_argument("--metadata", type=Path, required=True)
    updater.add_argument("--notes", type=Path, required=True)
    updater.add_argument("--output", type=Path, required=True)
    updater.set_defaults(func=write_updater)

    pypi = sub.add_parser("check-pypi")
    pypi.add_argument("--metadata", type=Path, required=True)
    pypi.add_argument("--version", required=True)
    pypi.add_argument("--github-output", type=Path)
    pypi.set_defaults(func=check_pypi)
    return root


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Atomically update the fixed vMLX website release surfaces on the origin."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def update_download_cards(html: str, downloads: dict, runtime: dict) -> str:
    """Keep visible checksums, sizes and OS requirements tied to the release."""
    floors = {
        "tahoe": runtime["platforms"]["macosx_26_0_arm64"]["minimum_system_version"],
        "sequoia": runtime["platforms"]["macosx_14_0_arm64"]["minimum_system_version"],
    }
    floors = {key: value.removesuffix(".0") for key, value in floors.items()}
    seen: set[str] = set()
    cards = re.compile(
        r'(<div class="build-h"><span>[^<]*</span><span>)([^<]*)'
        r'(</span></div>.*?<div class="sha"><em>sha256</em>)([0-9a-fA-F]{64})'
        r'(</div>\s*<a class="get" href=")([^"]+)(")', re.DOTALL,
    )

    def replace(match: re.Match) -> str:
        flavor = next((name for name, entry in downloads.items()
                       if entry["url"] == match[6]), None)
        if flavor not in floors or flavor in seen:
            raise SystemExit("ERROR: ambiguous download card artifact")
        entry = downloads[flavor]
        if (not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                or not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0):
            raise SystemExit(f"ERROR: invalid {flavor} checksum/size")
        seen.add(flavor)
        middle = match[3]
        middle = re.sub(r'(<div class="n">).*?(</div>)',
                        rf'\g<1>macOS {floors[flavor]}+\g<2>', middle, count=1)
        description = (f"Native MLX wheels for macOS {floors[flavor]} or later."
                       if flavor == "tahoe" else
                       f"Compatible with macOS {floors[flavor]} and later, including earlier Tahoe versions.")
        middle = re.sub(r'(<div class="d">).*?(</div>)',
                        lambda item: item[1] + description + item[2], middle, count=1)
        size = f"{entry['bytes'] / (1024 * 1024):.0f} MiB"
        return match[1] + size + middle + entry["sha256"] + match[5] + match[6] + match[7]

    html = cards.sub(replace, html)
    if seen != set(floors):
        raise SystemExit("ERROR: page must contain one checksum card per release flavor")
    guidance = (f"Choose the native build on macOS {floors['tahoe']} or later. "
                f"The compatibility build supports macOS {floors['sequoia']} or later.")
    html = re.sub(r'(<section id="builds".*?<p class="sec-lede">).*?(</p>)',
                  lambda item: item[1] + guidance + item[2], html, count=1, flags=re.DOTALL)
    return html


def atomic_write(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--updater", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--runtime-spec", type=Path, default=(
        Path(__file__).resolve().parents[2] / "panel/scripts/mlx-runtime.json"
    ))
    args = parser.parse_args()

    updater = json.loads(args.updater.read_text(encoding="utf-8"))
    version = str(updater.get("version", ""))
    if not VERSION_RE.fullmatch(version):
        raise SystemExit("ERROR: updater version is invalid")
    downloads = updater.get("downloads", {})
    tahoe = downloads.get("tahoe", {})
    sequoia = downloads.get("sequoia", {})
    for flavor, entry in (("tahoe", tahoe), ("sequoia", sequoia)):
        expected = f"vMLX-{version}-{flavor}-arm64.dmg"
        if not str(entry.get("url", "")).endswith(f"/v{version}/{expected}"):
            raise SystemExit(f"ERROR: invalid {flavor} updater URL")

    site_root = args.site_root.resolve()
    updater_target = site_root / "update" / "latest.json"
    download_page = site_root / "download" / "index.html"
    if not download_page.is_file():
        raise SystemExit(f"ERROR: missing download page: {download_page}")

    html = download_page.read_text(encoding="utf-8")
    matches = re.findall(r'"softwareVersion"\s*:\s*"([0-9]+\.[0-9]+\.[0-9]+)"', html)
    if len(matches) != 1:
        raise SystemExit("ERROR: download page must contain exactly one softwareVersion")
    old_version = matches[0]
    if VERSION_RE.fullmatch(old_version) is None:
        raise SystemExit("ERROR: current download page version is invalid")

    if old_version == version:
        if not updater_target.is_file():
            raise SystemExit("ERROR: current-version website is missing its updater")
        current_updater = json.loads(updater_target.read_text(encoding="utf-8"))
        if current_updater != updater:
            raise SystemExit("ERROR: current-version website updater differs from candidate")

    updated = html.replace(f"vMLX-{old_version}-", f"vMLX-{version}-")
    updated = updated.replace(f"/v{old_version}/", f"/v{version}/")
    updated = updated.replace(f'"softwareVersion": "{old_version}"', f'"softwareVersion": "{version}"')
    updated = re.sub(r'"fileSize"\s*:\s*"[0-9]+ bytes"',
                     f'"fileSize": "{tahoe.get("bytes", 0)} bytes"', updated, count=1)
    updated = re.sub(rf"(?<![0-9.]){re.escape(old_version)}(?![0-9]|\.[0-9])", version, updated)
    updated = update_download_cards(updated, downloads, json.loads(args.runtime_spec.read_text()))
    if old_version == version and updated == html:
        print(f"Website release surfaces already match {version}")
        return 0

    backup = args.backup_root.resolve() / f"vmlx-{old_version}-before-{version}"
    if backup.exists():
        backup_page = backup / "download-index.html"
        backup_updater = backup / "latest.json"
        if not backup_page.is_file() or backup_page.read_text(encoding="utf-8") != html:
            raise SystemExit(f"ERROR: existing release backup is not reusable: {backup}")
        if updater_target.is_file():
            if not backup_updater.is_file() or backup_updater.read_bytes() != updater_target.read_bytes():
                raise SystemExit(f"ERROR: existing updater backup is not reusable: {backup}")
    else:
        backup.mkdir(parents=True)
        if updater_target.is_file():
            shutil.copy2(updater_target, backup / "latest.json")
        shutil.copy2(download_page, backup / "download-index.html")

    for entry in (tahoe, sequoia):
        if entry["url"] not in updated:
            raise SystemExit(f"ERROR: patched page is missing {entry['url']}")
    if old_version != version and old_version in updated:
        raise SystemExit(f"ERROR: patched page still contains old version {old_version}")

    updater_bytes = (json.dumps(updater, indent=2) + "\n").encode("utf-8")
    atomic_write(updater_target, updater_bytes, 0o644)
    atomic_write(download_page, updated.encode("utf-8"), 0o644)
    print(f"Updated website release surfaces from {old_version} to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

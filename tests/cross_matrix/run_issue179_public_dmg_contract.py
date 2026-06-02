#!/usr/bin/env python3
"""Build a no-heavy MiniMax #179 public DMG server-route contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SERVER_REL = (
    "Contents/Resources/bundled-python/python/lib/python3.12/"
    "site-packages/vmlx_engine/server.py"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_app_server(mountpoint: Path) -> Path:
    candidates = sorted(mountpoint.glob(f"*.app/{SERVER_REL}"))
    if not candidates:
        candidates = sorted(mountpoint.glob(f"**/*.app/{SERVER_REL}"))
    if not candidates:
        raise FileNotFoundError(f"bundled vmlx_engine/server.py not found under {mountpoint}")
    return candidates[0]


def build_contract_from_mount(
    *,
    mountpoint: Path,
    dmg: Path,
    release_tag: str,
    asset: str,
    asset_size_bytes: int | None,
    out: Path,
) -> dict[str, Any]:
    server = find_app_server(mountpoint)
    text = read_text(server)
    route = '@app.post("/v1/responses/{response_id}/cancel"' in text
    abort = "async def cancel_response" in text and "await _engine.abort_request(response_id)" in text
    return {
        "artifact": str(out),
        "asset": asset,
        "asset_size_bytes": asset_size_bytes,
        "boundary": (
            f"Public {release_tag} {asset} DMG route/hash contract for issue #179 "
            "reporter bundle provenance; this does not load the model."
        ),
        "downloaded_dmg": str(dmg),
        "evidence_command": (
            f"hdiutil attach -readonly -nobrowse {dmg}; inspect bundled server.py"
        ),
        "mounted_server_path": str(server),
        "release_tag": release_tag,
        "server_cancel_calls_engine_abort": abort,
        "server_has_responses_cancel_route": route,
        "server_sha256": sha256_file(server),
    }


def attach_dmg(dmg: Path, mountpoint: Path) -> None:
    subprocess.run(
        [
            "hdiutil",
            "attach",
            "-readonly",
            "-nobrowse",
            "-mountpoint",
            str(mountpoint),
            str(dmg),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def detach_dmg(mountpoint: Path) -> None:
    subprocess.run(
        ["hdiutil", "detach", str(mountpoint), "-quiet"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def write_contract(
    *,
    dmg: Path,
    release_tag: str,
    asset: str,
    asset_size_bytes: int | None,
    out: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vmlx-issue179-dmg-") as tmp:
        mountpoint = Path(tmp) / "mount"
        mountpoint.mkdir()
        attached = False
        try:
            attach_dmg(dmg, mountpoint)
            attached = True
            contract = build_contract_from_mount(
                mountpoint=mountpoint,
                dmg=dmg,
                release_tag=release_tag,
                asset=asset,
                asset_size_bytes=asset_size_bytes,
                out=out,
            )
        finally:
            if attached:
                detach_dmg(mountpoint)
            shutil.rmtree(mountpoint, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmg", required=True, type=Path)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--asset-size-bytes", type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = write_contract(
        dmg=args.dmg,
        release_tag=args.release_tag,
        asset=args.asset,
        asset_size_bytes=args.asset_size_bytes,
        out=args.out,
    )
    print(json.dumps({"status": "pass", "out": str(args.out), **contract}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

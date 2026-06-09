#!/usr/bin/env python3
"""Audit current signed-checkpoint DMG readiness without publishing.

This runner intentionally does not build, sign, notarize, tag, upload, or
mutate updater feeds. It records whether existing local DMGs are already signed
and stapled, whether they can be treated as current for this source checkout,
and whether this process can fresh-sign with the release Developer ID identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-signed-checkpoint-dmg-readiness-20260609.json")
DEVELOPER_ID_NAME = "Developer ID Application: ShieldStack LLC (55KGF2S5AY)"
TEAM_ID = "55KGF2S5AY"
VERSIONED_DMGS = (
    "panel/release/vMLX-1.5.56-sequoia-arm64.dmg",
    "panel/release/vMLX-1.5.56-tahoe-arm64.dmg",
)


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "tail": proc.stdout.splitlines()[-60:],
    }


def git_head(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def git_head_time(root: Path) -> str:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%cI"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature_summary(path: Path) -> dict[str, Any]:
    result = run(["codesign", "-dv", "--verbose=4", str(path)])
    text = "\n".join(result["tail"])
    return {
        "returncode": result["returncode"],
        "authority_developer_id": DEVELOPER_ID_NAME in text,
        "team_identifier": TEAM_ID if f"TeamIdentifier={TEAM_ID}" in text else None,
        "signature_is_adhoc": "Signature=adhoc" in text,
        "tail": result["tail"],
    }


def audit_dmg(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    row: dict[str, Any] = {
        "path": rel,
        "exists": path.exists(),
        "sha256": sha256(path),
    }
    if not path.exists():
        return row
    row["mtime"] = path.stat().st_mtime
    row["hdiutil_verify"] = run(["hdiutil", "verify", str(path)])
    row["codesign_verify"] = run(["codesign", "--verify", "--verbose=2", str(path)])
    row["signature"] = signature_summary(path)
    row["stapler_validate"] = run(["xcrun", "stapler", "validate", str(path)])
    row["spctl_assess"] = run(
        [
            "spctl",
            "--assess",
            "--type",
            "open",
            "--context",
            "context:primary-signature",
            "--verbose=4",
            str(path),
        ]
    )
    row["signed_and_stapled"] = (
        row["hdiutil_verify"]["returncode"] == 0
        and row["codesign_verify"]["returncode"] == 0
        and row["signature"]["authority_developer_id"]
        and row["signature"]["team_identifier"] == TEAM_ID
        and not row["signature"]["signature_is_adhoc"]
        and row["stapler_validate"]["returncode"] == 0
        and row["spctl_assess"]["returncode"] == 0
    )
    return row


def fresh_signing_probe() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vmlx-current-signing-probe.") as tmp:
        probe = Path(tmp) / "codesign-probe"
        probe.write_bytes(Path("/bin/echo").read_bytes())
        probe.chmod(0o755)
        sign = run(
            [
                "codesign",
                "--force",
                "--timestamp",
                "--options",
                "runtime",
                "--sign",
                DEVELOPER_ID_NAME,
                str(probe),
            ]
        )
        display = signature_summary(probe)
    return {
        "identity": DEVELOPER_ID_NAME,
        "status": "pass" if sign["returncode"] == 0 and display["authority_developer_id"] else "blocked",
        "sign": sign,
        "signature": display,
    }


def current_installed_app() -> dict[str, Any]:
    path = Path("/Applications/vMLX.app")
    row: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        verify = run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(path)])
        sig = signature_summary(path)
        row.update(
            {
                "codesign_verify": verify,
                "signature_is_adhoc": sig["signature_is_adhoc"],
                "developer_id_signed": sig["authority_developer_id"],
                "team_identifier": sig["team_identifier"],
                "codesign_valid": verify["returncode"] == 0,
                "signature": sig,
            }
        )
    return row


def keychain_status() -> list[dict[str, Any]]:
    rows = []
    for path in (
        Path("~/Library/Keychains/vmlx-build.keychain-db").expanduser(),
        Path("~/Library/Keychains/build.keychain-db").expanduser(),
        Path("~/Library/Keychains/login.keychain-db").expanduser(),
    ):
        info = run(["security", "show-keychain-info", str(path)])
        rows.append(
            {
                "path": str(path),
                "returncode": info["returncode"],
                "tail": info["tail"],
                "locked_or_interactive_blocked": info["returncode"] != 0,
            }
        )
    return rows


def notary_status() -> dict[str, Any]:
    keychain = str(Path("~/Library/Keychains/vmlx-build.keychain-db").expanduser())
    history = run(
        [
            "xcrun",
            "notarytool",
            "history",
            "--keychain",
            keychain,
            "--keychain-profile",
            "vmlx-notary",
            "--output-format",
            "json",
        ]
    )
    return {
        "required_profile": "vmlx-notary",
        "required_keychain": keychain,
        "history": history,
        "status": "pass" if history["returncode"] == 0 else "blocked",
    }


def build_audit(root: Path, out: Path = DEFAULT_OUT) -> dict[str, Any]:
    root = root.resolve()
    head = git_head(root)
    dmgs = [audit_dmg(root, rel) for rel in VERSIONED_DMGS]
    signing = fresh_signing_probe()
    notary = notary_status()
    installed = current_installed_app()
    keychains = keychain_status()
    all_existing_dmgs_signed = bool(dmgs) and all(row.get("signed_and_stapled") for row in dmgs)
    fresh_release_ready = (
        signing["status"] == "pass"
        and notary["status"] == "pass"
        and all_existing_dmgs_signed
        and not installed.get("signature_is_adhoc", True)
    )
    result = {
        "artifact": str(out),
        "head": head,
        "head_time": git_head_time(root),
        "status": "pass" if fresh_release_ready else "open",
        "existing_dmgs": dmgs,
        "existing_dmgs_signed_and_stapled": all_existing_dmgs_signed,
        "existing_dmgs_current_for_head": False,
        "current_installed_app": installed,
        "fresh_signing_probe": signing,
        "keychains": keychains,
        "notarization": notary,
        "required_next_steps": [
            "unlock_vmlx_build_keychain",
            "restore_codesign_partition_list",
            "rerun_fresh_developer_id_signing_probe",
            "rebuild_current_source_dmg_flavors",
            "notarize_staple_and_verify_current_dmgs",
        ],
        "release_boundary": (
            "Existing local DMGs may be signed/stapled, but they are not current-source "
            "checkpoint proof unless rebuilt after this HEAD and verified again."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_audit(Path("."), args.out)
    print(args.out)
    print(f"status={result['status']}")
    print(f"existing_dmgs_signed_and_stapled={result['existing_dmgs_signed_and_stapled']}")
    print(f"fresh_signing_probe={result['fresh_signing_probe']['status']}")
    print(f"notarization={result['notarization']['status']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

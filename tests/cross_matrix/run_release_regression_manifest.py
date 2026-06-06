#!/usr/bin/env python3
"""Write the release-regression manifest artifact.

This command does not run the heavy suite. It produces a stable JSON inventory
of the release-regression rows and the commands/artifacts that prove each row.
Use it before expanding or auditing the suite so new fixes land in a named row
instead of another one-off command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cross_matrix.release_regression_manifest import (
    DEFERRED_RELEASE_OPEN_REQUIREMENTS,
    build_manifest,
    validate_current_proof_sweep_artifacts,
)


DEFAULT_OUT = Path(
    "build/current-release-regression-manifest-after-cross-family-smoke-refresh-20260606.json"
)
PREPACKAGE_ALLOWED_BLOCKERS = {
    "packaged_app_developer_id_signing_blocked",
    "installed_app_runtime_parity_audit",
}


def release_clearance_from_proof_sweep(current_proof_sweep: dict) -> dict:
    regression_suite = current_proof_sweep.get("regression_suite") or {}
    open_requirements = [
        str(item) for item in regression_suite.get("open_requirements") or []
    ]
    effective_open_requirements = [
        item for item in open_requirements if item not in DEFERRED_RELEASE_OPEN_REQUIREMENTS
    ]
    blocker_ledger = current_proof_sweep.get("release_blocker_ledger") or {}
    blockers = [
        item for item in blocker_ledger.get("blockers") or [] if isinstance(item, dict)
    ]
    proof_sweep_status = str(current_proof_sweep.get("status"))
    proof_sweep_failed_components = [
        str(item) for item in current_proof_sweep.get("failed_components") or []
    ]
    proof_sweep_failure_is_deferred = (
        proof_sweep_status == "pass"
        or (
            proof_sweep_status == "fail"
            and set(proof_sweep_failed_components) <= {"regression_suite"}
            and not effective_open_requirements
        )
    )
    release_ready = proof_sweep_failure_is_deferred and not effective_open_requirements and not blockers
    return {
        "status": "pass" if release_ready else "open",
        "release_ready": release_ready,
        "proof_sweep_status": proof_sweep_status,
        "proof_sweep_failed_components": proof_sweep_failed_components,
        "proof_sweep_failure_is_deferred": proof_sweep_failure_is_deferred,
        "open_requirements": open_requirements,
        "effective_open_requirements": effective_open_requirements,
        "deferred_open_requirements": [
            item for item in open_requirements if item in DEFERRED_RELEASE_OPEN_REQUIREMENTS
        ],
        "blockers": blockers,
        "reason": (
            "All current proof-sweep artifacts pass and no release blockers remain."
            if release_ready
            else "Release is not cleared while non-deferred open requirements or release blockers remain."
        ),
    }


def prepackage_clearance_from_release_clearance(release_clearance: dict) -> dict:
    open_requirements = [
        str(item) for item in release_clearance.get("effective_open_requirements") or []
    ]
    blockers = [
        item
        for item in release_clearance.get("blockers") or []
        if isinstance(item, dict)
    ]
    blocking_before_package = [
        item
        for item in blockers
        if str(item.get("id")) not in PREPACKAGE_ALLOWED_BLOCKERS
    ]
    proof_sweep_status = str(release_clearance.get("proof_sweep_status"))
    proof_sweep_failed_components = [
        str(item) for item in release_clearance.get("proof_sweep_failed_components") or []
    ]
    proof_sweep_prepackage_ok = proof_sweep_status == "pass" or (
        set(proof_sweep_failed_components)
        <= {
            "no_not_pass_post_budget_artifacts",
            "no_release_blockers",
            "no_open_objective_requirements",
            "regression_suite",
            "packaged_integrity_matrix",
            "installed_app_runtime_parity_audit",
            "staged_app_runtime_parity_audit",
            "public_app_issue_audit",
        }
    )
    prepackage_ready = (
        proof_sweep_prepackage_ok
        and not open_requirements
        and not blocking_before_package
    )
    return {
        "status": "pass" if prepackage_ready else "open",
        "prepackage_ready": prepackage_ready,
        "proof_sweep_status": proof_sweep_status,
        "proof_sweep_failed_components": proof_sweep_failed_components,
        "open_requirements": open_requirements,
        "blocking_before_package": blocking_before_package,
        "allowed_packaging_blockers": [
            item
            for item in blockers
            if str(item.get("id")) in PREPACKAGE_ALLOWED_BLOCKERS
        ],
        "reason": (
            "All non-packaging blockers are cleared; building signed artifacts may proceed."
            if prepackage_ready
            else "Pre-package build is not cleared while model/proof blockers remain."
        ),
    }


def build_manifest_artifact(
    root: Path,
    *,
    require_current_proof_sweep: bool = False,
    require_release_ready: bool = False,
    require_prepackage_ready: bool = False,
) -> dict:
    manifest = build_manifest()
    manifest["current_proof_sweep"] = validate_current_proof_sweep_artifacts(root)
    manifest["release_clearance"] = release_clearance_from_proof_sweep(
        manifest["current_proof_sweep"]
    )
    manifest["prepackage_clearance"] = prepackage_clearance_from_release_clearance(
        manifest["release_clearance"]
    )
    manifest["release_ready"] = bool(manifest["release_clearance"]["release_ready"])
    manifest["prepackage_ready"] = bool(
        manifest["prepackage_clearance"]["prepackage_ready"]
    )
    manifest["release_blockers"] = list(manifest["release_clearance"]["blockers"])
    proof_sweep_failed = manifest["current_proof_sweep"]["status"] != "pass"
    proof_sweep_failure_allowed_for_release = (
        require_release_ready
        and manifest["release_ready"]
    )
    proof_sweep_failure_allowed_for_prepackage = (
        require_prepackage_ready
        and not require_current_proof_sweep
        and not require_release_ready
        and manifest["prepackage_ready"]
    )
    release_not_ready = require_release_ready and not manifest["release_ready"]
    prepackage_not_ready = (
        require_prepackage_ready and not manifest["prepackage_ready"]
    )
    manifest["status"] = (
        "fail"
        if (
            (proof_sweep_failed and not proof_sweep_failure_allowed_for_prepackage)
            and not proof_sweep_failure_allowed_for_release
            or release_not_ready
            or prepackage_not_ready
        )
        else "pass"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--require-current-proof-sweep",
        action="store_true",
        help="Exit nonzero unless every current post-budget-edge proof artifact exists and has status=pass.",
    )
    parser.add_argument(
        "--require-release-ready",
        action="store_true",
        help="Exit nonzero unless current proof sweep passes and no release blockers/open requirements remain.",
    )
    parser.add_argument(
        "--require-prepackage-ready",
        action="store_true",
        help="Exit nonzero unless current proof sweep passes and only packaging/signing blockers remain.",
    )
    args = parser.parse_args()

    manifest = build_manifest_artifact(
        Path("."),
        require_current_proof_sweep=args.require_current_proof_sweep,
        require_release_ready=args.require_release_ready,
        require_prepackage_ready=args.require_prepackage_ready,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"rows={len(manifest['rows'])}")
    print("domains=" + ",".join(sorted({row["domain"] for row in manifest["rows"]})))
    print(f"current_proof_sweep={manifest['current_proof_sweep']['status']}")
    print(f"prepackage_ready={str(manifest['prepackage_ready']).lower()}")
    print(f"release_ready={str(manifest['release_ready']).lower()}")
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
    build_manifest,
    validate_current_proof_sweep_artifacts,
)


DEFAULT_OUT = Path("build/current-release-regression-manifest-20260521.json")


def build_manifest_artifact(
    root: Path,
    *,
    require_current_proof_sweep: bool = False,
) -> dict:
    manifest = build_manifest()
    manifest["current_proof_sweep"] = validate_current_proof_sweep_artifacts(root)
    manifest["status"] = (
        "fail"
        if require_current_proof_sweep
        and manifest["current_proof_sweep"]["status"] != "pass"
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
    args = parser.parse_args()

    manifest = build_manifest_artifact(
        Path("."),
        require_current_proof_sweep=args.require_current_proof_sweep,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"rows={len(manifest['rows'])}")
    print("domains=" + ",".join(sorted({row["domain"] for row in manifest["rows"]})))
    print(f"current_proof_sweep={manifest['current_proof_sweep']['status']}")
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

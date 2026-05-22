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

from tests.cross_matrix.release_regression_manifest import build_manifest


DEFAULT_OUT = Path("build/current-release-regression-manifest-20260521.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"rows={len(manifest['rows'])}")
    print("domains=" + ",".join(sorted({row["domain"] for row in manifest["rows"]})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

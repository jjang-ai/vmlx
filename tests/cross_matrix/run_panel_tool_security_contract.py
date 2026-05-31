#!/usr/bin/env python3
"""Run panel tool-loop/security contracts from the correct panel cwd."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json")


def run_contract(root: Path) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        [
            "npx",
            "vitest",
            "run",
            "tests/tool-executor-security.test.ts",
            "tests/tool-auto-continue.test.ts",
        ],
        cwd=root / "panel",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if proc.returncode == 0 else "open",
        "command": [
            "cd",
            "panel",
            "&&",
            "npx",
            "vitest",
            "run",
            "tests/tool-executor-security.test.ts",
            "tests/tool-auto-continue.test.ts",
        ],
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = run_contract(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"returncode={artifact['returncode']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

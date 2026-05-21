#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vmlx_engine.native_mtp_examples.mtp_runtime_common import inspect_bundle_no_load, print_json


def inspect_path(model_dir: Path) -> dict:
    status = inspect_bundle_no_load(model_dir)
    return {
        "path": str(model_dir),
        "no_model_load": True,
        "status": status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect native MTP metadata without loading model weights."
    )
    parser.add_argument("model_dir", type=Path, help="Model bundle directory")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args(argv)

    report = inspect_path(args.model_dir)
    status = report["status"]
    if args.json:
        print_json(report)
    else:
        print(f"Native MTP metadata: {report['path']}")
        print("No model weights loaded.")
        print(f"status={status.get('status')} family={status.get('family')}")
        print(
            "artifact_available="
            f"{status.get('artifact_available')} runtime_supported="
            f"{status.get('runtime_supported')} runtime_available="
            f"{status.get('runtime_available')}"
        )
        print(
            "layers="
            f"{status.get('config_num_nextn_predict_layers')} tensors="
            f"{status.get('mtp_tensor_count')} depth="
            f"{status.get('effective_depth')} source="
            f"{status.get('effective_depth_source')}"
        )
        print(f"reason={status.get('runtime_reason')}")
        for issue in status.get("issues") or []:
            print(f"issue: {issue}")
    return 0 if not status.get("issues") else 1


if __name__ == "__main__":
    raise SystemExit(main())

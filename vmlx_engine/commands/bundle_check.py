"""CLI wrapper for the shared model-bundle integrity preflight."""

from __future__ import annotations

import json


def bundle_check_command(args) -> None:
    from ..model_bundle_integrity import BundleIntegrityError, check_model_bundle

    try:
        result = check_model_bundle(
            args.model,
            repair=not args.no_repair,
            use_cache=not args.no_cache,
        )
    except BundleIntegrityError as exc:
        result = {
            "schema": "vmlx-model-bundle-integrity-v1",
            "status": "error",
            "bundle": str(args.model),
            "error": str(exc),
        }
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"Bundle integrity: ERROR\n{exc}")
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps(result, sort_keys=True))
        return
    source = "cached one-time stamp" if result["cache_hit"] else "fresh header scan"
    print(f"Bundle integrity: OK ({source})")
    print(
        f"  shards={result['shards']} tensors={result['tensors']} "
        f"remaining_misaligned={result['misaligned_tensors']}"
    )
    for repaired in result["repairs"]:
        print(f"  atomically repaired: {repaired}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")


__all__ = ["bundle_check_command"]

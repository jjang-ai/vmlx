#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vmlx_engine.native_mtp_examples.mtp_runtime_common import command_matrix_rows, print_json


def parse_depths(raw: str) -> list[int]:
    depths: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            depths.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"depth values must be integers, got {item!r}"
            ) from exc
    return depths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate native MTP depth/env dry-run command matrix."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--no-default", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="native-mtp-local")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        depths = parse_depths(args.depths)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    rows = command_matrix_rows(
        args.model_dir,
        depths=depths,
        include_disabled=args.include_disabled,
        include_default=not args.no_default,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )
    report = {"no_model_load": True, "rows": rows}
    if args.json:
        print_json(report)
    else:
        for row in rows:
            print(f"[{row['label']}] depth={row.get('depth')} source={row.get('depth_source')}")
            print(row["shell"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

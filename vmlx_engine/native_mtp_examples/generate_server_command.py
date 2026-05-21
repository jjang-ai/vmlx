#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vmlx_engine.native_mtp_examples.mtp_runtime_common import build_server_command, print_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a dry-run native MTP vmlx serve command."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--depth", type=int, default=None, help="Native MTP depth, clamped to 1..3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="native-mtp-local")
    parser.add_argument("--mllm", action="store_true", help="Add --is-mllm to the dry-run command")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    plan = build_server_command(
        args.model_dir,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        depth=args.depth,
        mllm=args.mllm,
        disable_prefix_cache=args.disable_prefix_cache,
    )
    if args.json:
        print_json(plan)
    else:
        print(plan["warning"])
        print("Dry-run only; command was not executed.")
        print(plan["shell"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from vmlx_engine.native_mtp_examples.mtp_runtime_common import (
    LIVE_RUN_WARNING,
    build_server_command,
    print_json,
    require_live_run_allowed,
    shell_join,
)


def build_smoke_plan(
    model_dir: str | Path,
    *,
    depth: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "native-mtp-local",
    mllm: bool = False,
) -> dict:
    server = build_server_command(
        model_dir,
        host=host,
        port=port,
        model_name=model_name,
        depth=depth,
        mllm=mllm,
    )
    base_url = f"http://{host}:{port}"
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Count 1, 2, 3."}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        separators=(",", ":"),
    )
    return {
        "dry_run": True,
        "no_model_load": True,
        "warning": LIVE_RUN_WARNING,
        "server": server,
        "checks": [
            ["curl", "-fsS", f"{base_url}/health"],
            [
                "curl",
                "-fsS",
                f"{base_url}/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
            ],
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or explicitly execute a guarded native MTP server smoke plan."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="native-mtp-local")
    parser.add_argument("--mllm", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Launch server; guarded and off by default")
    parser.add_argument("--allow-live", action="store_true", help="Required with the live-run acknowledgement env")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    plan = build_smoke_plan(
        args.model_dir,
        depth=args.depth,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        mllm=args.mllm,
    )
    if not args.execute:
        if args.json:
            print_json(plan)
        else:
            print(plan["warning"])
            print("Dry-run only; server was not launched.")
            print(plan["server"]["shell"])
            for check in plan["checks"]:
                print(shell_join(check))
        return 0

    require_live_run_allowed(allow_live=args.allow_live)
    proc_env = dict(os.environ)
    proc_env.update(plan["server"]["env"])
    proc = subprocess.Popen(plan["server"]["command"], env=proc_env)
    print(f"Started server pid={proc.pid}; stop it manually when smoke checks finish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for vmlx-worker.

Usage:
    # Preferred: secret via env var (so it doesn't land in `ps aux`)
    export VMLX_CLUSTER_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    vmlx-worker --port 9100

    # LAN deployment requires --allow-public to bind to all interfaces
    vmlx-worker --port 9100 --allow-public
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from .discovery import DEFAULT_WORKER_PORT


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vmlx-worker",
        description=(
            "vMLX distributed inference worker — runs on secondary Macs to "
            "share compute. Experimental: localhost loopback testing only "
            "is recommended at this time. See docs/guides/distributed-setup.md."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_WORKER_PORT,
        help=f"Port to listen on (default: {DEFAULT_WORKER_PORT})",
    )
    parser.add_argument(
        "--secret", type=str, default=None,
        help=(
            "Cluster secret for authentication (must match coordinator). "
            "Prefer setting VMLX_CLUSTER_SECRET env var instead so the secret "
            "doesn't appear in `ps aux` process listings."
        ),
    )
    parser.add_argument(
        "--bind", type=str, default="127.0.0.1",
        help=(
            "Interface to bind to (default: 127.0.0.1 for localhost-only "
            "testing). Use --allow-public for LAN/Thunderbolt/Tailscale."
        ),
    )
    parser.add_argument(
        "--allow-public", action="store_true",
        help=(
            "Bind to 0.0.0.0 (all interfaces) instead of --bind. "
            "Required for multi-Mac clusters. Logs a warning — only use on "
            "trusted networks (wired LAN, Thunderbolt bridge, or Tailscale)."
        ),
    )
    parser.add_argument(
        "--no-advertise", action="store_true",
        help="Disable Bonjour/mDNS advertisement (use manual IP on coordinator)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--allowed-model-root", type=str, action="append", default=None,
        help=(
            "Directory path that this worker is allowed to load models from. "
            "May be specified multiple times. Defaults to ~/mlx, ~/.cache/huggingface, "
            "~/.cache/vmlx, and the current working directory. This allowlist "
            "prevents a compromised coordinator from instructing the worker to "
            "read arbitrary files off disk."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("vmlx-worker")

    # Secret precedence: CLI arg > env var. Refuse to start with an empty
    # secret — before this fix an empty secret silently accepted any
    # incoming JOIN, which is worse than not running at all.
    secret = args.secret if args.secret is not None else os.environ.get(
        "VMLX_CLUSTER_SECRET", "",
    )
    if not secret:
        logger.error(
            "Refusing to start: cluster secret is required. Set "
            "VMLX_CLUSTER_SECRET environment variable (preferred) or pass "
            "--secret. Generate a random one with: "
            "python3 -c 'import secrets; print(secrets.token_urlsafe(24))'"
        )
        sys.exit(2)

    bind = "0.0.0.0" if args.allow_public else args.bind

    from .worker import Worker
    from .discovery import get_local_node_info

    node = get_local_node_info()
    logger.info("=" * 60)
    logger.info("vMLX Worker (EXPERIMENTAL — localhost testing recommended)")
    logger.info("=" * 60)
    logger.info("  Host: %s", node.hostname)
    logger.info("  Chip: %s", node.chip)
    logger.info("  RAM:  %d GB (%d GB available)", node.ram_gb, int(node.available_gb))
    logger.info("  Bind: %s:%d", bind, args.port)
    logger.info("  Bonjour: %s", "OFF" if args.no_advertise else "ON")
    logger.info("  Auth: cluster secret set (%d bytes)", len(secret))
    if args.allow_public:
        logger.warning(
            "  --allow-public: binding to 0.0.0.0. Worker is reachable from "
            "every network interface. Only do this on trusted networks."
        )
    logger.info("=" * 60)

    try:
        worker = Worker(
            port=args.port,
            cluster_secret=secret,
            advertise=not args.no_advertise,
            bind=bind,
            allowed_model_roots=args.allowed_model_root,
        )
    except ValueError as e:
        logger.error("Worker refused to start: %s", e)
        sys.exit(2)

    loop = asyncio.new_event_loop()

    # Signal-safe shutdown. The previous implementation did
    # `loop.create_task(worker.shutdown())` then immediately
    # `loop.call_soon(loop.stop)` — the task never ran because the loop
    # stopped before its next tick, leaving resources leaked on signal.
    # The fix: set an asyncio.Event; the main coroutine awaits both the
    # serve loop and the event so Ctrl-C cleanly unwinds via asyncio.
    stop_event = asyncio.Event()

    def _signal_handler(sig: int, _frame) -> None:
        logger.info("Received signal %d, shutting down...", sig)
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    async def _run() -> None:
        serve_task = asyncio.create_task(worker.serve())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {serve_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        # Surface any serve() exception instead of swallowing it silently.
        if serve_task in done and serve_task.exception() is not None:
            logger.error("Worker serve() crashed: %s", serve_task.exception())

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        try:
            loop.run_until_complete(worker.shutdown())
        except Exception as e:
            logger.debug("Shutdown error: %s", e)
        loop.close()


if __name__ == "__main__":
    main()

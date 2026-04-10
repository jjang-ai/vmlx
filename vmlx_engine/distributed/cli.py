# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for vmlx-worker.

Usage:
    vmlx-worker --port 9100 --secret my-cluster-secret
    vmlx-worker --port 9100 --secret my-cluster-secret --no-advertise
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from .discovery import DEFAULT_WORKER_PORT


def main():
    parser = argparse.ArgumentParser(
        prog="vmlx-worker",
        description="vMLX distributed inference worker — runs on secondary Macs to share compute",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_WORKER_PORT,
        help=f"Port to listen on (default: {DEFAULT_WORKER_PORT})",
    )
    parser.add_argument(
        "--secret", type=str, default="",
        help="Cluster secret for authentication (must match coordinator)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("vmlx-worker")

    from .worker import Worker
    from .discovery import get_local_node_info

    node = get_local_node_info()
    logger.info("=" * 60)
    logger.info("vMLX Worker")
    logger.info("=" * 60)
    logger.info("  Host: %s", node.hostname)
    logger.info("  Chip: %s", node.chip)
    logger.info("  RAM:  %d GB (%d GB available)", node.ram_gb, int(node.available_gb))
    logger.info("  Port: %d", args.port)
    logger.info("  Bonjour: %s", "OFF" if args.no_advertise else "ON")
    logger.info("  Auth: %s", "cluster secret set" if args.secret else "NO SECRET (insecure)")
    logger.info("=" * 60)

    if not args.secret:
        logger.warning(
            "No cluster secret set! Any device on your network can "
            "connect and use this worker. Use --secret to require authentication."
        )

    worker = Worker(
        port=args.port,
        cluster_secret=args.secret,
        advertise=not args.no_advertise,
    )

    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        logger.info("Received signal %s, shutting down...", sig)
        loop.create_task(worker.shutdown())
        loop.call_soon(loop.stop)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(worker.serve())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        loop.run_until_complete(worker.shutdown())
        loop.close()


if __name__ == "__main__":
    main()

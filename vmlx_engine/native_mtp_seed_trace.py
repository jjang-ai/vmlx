"""Opt-in host wall intervals for native-MTP seed/re-entry diagnosis.

These intervals deliberately add no MLX evaluation or synchronization. A
later wait may include GPU work enqueued by an earlier interval, so the
numbers must not be presented as isolated kernel execution times.
"""

import json
import os
import time


class NativeMTPSeedTrace:
    def __init__(self):
        self.started = self.previous = time.perf_counter()
        self.stages_ms = {}

    def mark(self, name):
        now = time.perf_counter()
        self.stages_ms[name] = (now - self.previous) * 1000.0
        self.previous = now

    def emit(self, logger, **metadata):
        logger.info(
            "MLLM native MTP seed stages %s",
            json.dumps({
                **metadata,
                "stages_ms": self.stages_ms,
                "total_ms": (self.previous - self.started) * 1000.0,
                "clock": "host_wall_no_added_sync",
                "async_work_may_be_charged_to_later_waits": True,
            }, separators=(",", ":"), allow_nan=False),
        )


def start_native_mtp_seed_trace():
    enabled = os.environ.get("VMLINUX_NATIVE_MTP_SEED_TRACE") or os.environ.get(
        "VMLX_NATIVE_MTP_SEED_TRACE", ""
    )
    if enabled.lower() not in {"1", "true", "yes", "on"}:
        return None
    return NativeMTPSeedTrace()

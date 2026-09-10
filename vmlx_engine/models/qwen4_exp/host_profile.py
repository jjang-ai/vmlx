"""Bounded opt-in host attribution; never inserts a GPU evaluation fence."""

import cProfile
import functools
import io
import logging
import os
import pstats
import threading

logger = logging.getLogger(__name__)


def profile_decode_forward(function):
    """Profile the first 32 single-row calls per thread when explicitly enabled.

    With the flag absent, return the original function: no wrapper or hot-path
    environment lookup. Timings include Python/native host calls and any waits
    those calls already perform; they are not GPU kernel timings or throughput.
    """
    if os.environ.get("VMLX_QWEN4_HOST_PROFILE") != "1":
        return function
    local = threading.local()

    @functools.wraps(function)
    def wrapped(self, inputs, *args, **kwargs):
        count = getattr(local, "count", 0)
        if count >= 32 or tuple(getattr(inputs, "shape", ())) != (1, 1):
            return function(self, inputs, *args, **kwargs)
        if count == 0:
            local.profiler = cProfile.Profile()
        local.count = count + 1
        local.profiler.enable()
        try:
            return function(self, inputs, *args, **kwargs)
        finally:
            local.profiler.disable()
            if local.count == 32:
                report = io.StringIO()
                stats = pstats.Stats(local.profiler, stream=report).strip_dirs()
                stats.sort_stats("tottime").print_stats(40)
                stats.sort_stats("cumulative").print_stats(30)
                logger.info(
                    "QWEN4_HOST_PROFILE calls=32 scope=forward_host_only "
                    "instrumented=true gpu_fences_added=false\n%s",
                    report.getvalue(),
                )
                del local.profiler

    return wrapped

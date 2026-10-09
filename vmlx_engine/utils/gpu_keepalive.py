"""GPU keep-alive while a large model is resident (vMLX Python, INTERNAL, 2026-10-09).

Measured on M5 Max with Naive-N0.5 B97 (95.9 GiB resident through the loader's wired limit): after >= 1.5 s of GPU
idle, the NEXT GPU call pays a 0.65-1.25 s wake penalty (a 42 ms layer took 1.2 s); with no large resident model the
same idle costs only the clock ramp (~+30-55%). A tiny GPU op every <= 1 s removes the penalty completely (42-43 ms
after 1.5/3/5 s idle). Cost: one 256-element add per period on a private stream.

Usage: KeepAlive(period_s=0.5, idle_timeout_s=120).start(); call .touch() on every request so the beat stops after
idle_timeout_s without traffic (no power drain on an abandoned server)."""
from __future__ import annotations

import threading
import time

import mlx.core as mx


class KeepAlive:
    def __init__(self, period_s: float = 0.5, idle_timeout_s: float = 120.0):
        self.period_s, self.idle_timeout_s = period_s, idle_timeout_s
        self._stop = threading.Event(); self._last = time.monotonic(); self._th = None; self.beats = 0

    def touch(self):
        self._last = time.monotonic()
        if self._th is None or not self._th.is_alive():
            self.start()

    def start(self):
        self._stop.clear(); self._last = time.monotonic()
        self._th = threading.Thread(target=self._run, name="vmlx-gpu-keepalive", daemon=True); self._th.start()
        return self

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join()

    def _run(self):
        s = mx.new_stream(mx.gpu)                     # streams are per-thread: create it here
        x = mx.zeros((256,), stream=s)
        while not self._stop.is_set():
            if time.monotonic() - self._last > self.idle_timeout_s:
                return                                # no traffic: let the GPU sleep; touch() restarts
            with mx.stream(s):
                mx.eval(x + 1)
            self.beats += 1
            self._stop.wait(self.period_s)


# ---------------------------------------------------------------------------------------------- process-wide singleton
_GLOBAL: KeepAlive | None = None


def keepalive_enabled() -> bool:
    import os
    return os.environ.get("VMLX_GPU_KEEPALIVE", "1").strip().lower() not in {"0", "false", "no", "off"}


def touch_global():
    """Start (or refresh) the process-wide keep-alive. Cheap: a monotonic read + a thread liveness check.
    Env: VMLX_GPU_KEEPALIVE=0 disables; VMLX_GPU_KEEPALIVE_IDLE_S (default 600) stops the beat after that long
    without a touch, so an idle server lets the GPU sleep."""
    global _GLOBAL
    if not keepalive_enabled():
        return
    if _GLOBAL is None:
        import os
        _GLOBAL = KeepAlive(period_s=0.5, idle_timeout_s=float(os.environ.get("VMLX_GPU_KEEPALIVE_IDLE_S", "600")))
    _GLOBAL.touch()


def stop_global():
    """Release the owned heartbeat on model unload; an idle app may sleep."""
    global _GLOBAL
    instance, _GLOBAL = _GLOBAL, None
    if instance is not None:
        instance.stop()

"""Last-request telemetry for the Perf panel: one producer for every lane.

The Perf / Cache panels read ``/health`` and ``/v1/cache/stats``. Per-request decode tok/s, prefill tok/s (pp/s),
TTFT and speculative-decoding acceptance existed only as chat-stream usage receipts (``usage.vmlx_decode`` /
``usage.vmlx_prefill``) or in log lines, so the panels could not show them, and every per-lane edit risked
breaking whatever partial wiring existed (T-PERF, 2026-10-11).

``instrument(engine)`` wraps the loaded engine's four public generation methods once (server.load_model). Every API
route and every lane (batched text / MLLM, SimpleEngine, both DFlash2 lanes) goes through them, so this is the single
place the record is built:

    prompt / cached / completion tokens, cache tier, TTFT (method entry -> first produced token),
    prefill receipt (engine-measured, from ``output.prefill_usage``), cached pp/s (whole prompt over TTFT on a hit),
    decode tokens/seconds/tps (producer timestamps ``output.generated_at``, as the chat footer's decode receipt),
    speculative stats when the lane attaches them (``output.spec_stats``: DFlash2 acceptance, emitted per cycle,
    verify widths, AR-burst tokens).

``last()`` returns a copy of the newest record (the last request to finish or be aborted).
"""
from __future__ import annotations

import functools
import math
import threading
import time
from typing import Any, Optional

_LOCK = threading.Lock()
_LAST: Optional[dict] = None
_SEQ = 0


def _produced_at(output: Any) -> float:
    """Producer timestamp (perf_counter clock), else observation time: same rule as the decode usage receipt."""
    ts = getattr(output, "generated_at", None)
    if isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(ts) and ts > 0:
        return float(ts)
    return time.perf_counter()


def _rate(tokens: int, seconds: float) -> Optional[float]:
    return round(tokens / seconds, 1) if tokens > 0 and seconds > 0 else None


class _Tracker:
    __slots__ = ("method", "engine_type", "model_type", "request_id", "start", "first_ts", "first_count", "last_ts", "last_count",
                 "prompt", "cached", "detail", "completion", "prefill", "spec", "finish", "outputs")

    def __init__(self, method: str, engine_type: str, model_type: str, request_id: str | None = None):
        self.method, self.engine_type, self.model_type = method, engine_type, model_type
        self.request_id = request_id
        self.start = time.perf_counter()
        self.first_ts = self.last_ts = None
        self.first_count = self.last_count = 0
        self.prompt = self.cached = self.completion = 0
        self.detail = ""
        self.prefill = self.spec = self.finish = None
        self.outputs = 0

    def observe(self, output: Any) -> None:
        self.outputs += 1
        prompt = int(getattr(output, "prompt_tokens", 0) or 0)
        if prompt:
            self.prompt = prompt
        cached = int(getattr(output, "cached_tokens", 0) or 0)
        if cached:
            self.cached = cached
        detail = getattr(output, "cache_detail", "") or ""
        if detail:
            self.detail = str(detail)
        completion = int(getattr(output, "completion_tokens", 0) or 0)
        if completion > self.last_count:
            ts = _produced_at(output)
            if self.first_ts is None:
                self.first_ts, self.first_count = ts, completion
            self.last_ts, self.last_count = ts, completion
        self.completion = max(self.completion, completion)
        prefill = getattr(output, "prefill_usage", None)
        if isinstance(prefill, dict):
            self.prefill = prefill
        spec = getattr(output, "spec_stats", None)
        if isinstance(spec, dict):
            self.spec = spec
        if getattr(output, "finished", False):
            self.finish = getattr(output, "finish_reason", None) or "stop"

    def record(self, aborted: bool = False) -> None:
        if self.outputs == 0:
            return
        ttft = (self.first_ts - self.start) if self.first_ts is not None else None
        prefill = None
        if self.prefill is not None:
            tokens = self.prefill.get("tokens")
            seconds = self.prefill.get("seconds")
            if isinstance(tokens, int) and isinstance(seconds, (int, float)):
                prefill = {
                    "tokens": tokens,
                    "seconds": round(float(seconds), 4),
                    "tps": _rate(tokens, float(seconds)),
                    "scope": self.prefill.get("scope"),
                    "path": self.prefill.get("path"),
                }
        decode = None
        if self.first_ts is not None and self.last_ts is not None and self.last_count > self.first_count:
            d_tokens = self.last_count - self.first_count
            d_seconds = self.last_ts - self.first_ts
            if d_seconds > 0:
                decode = {"tokens": d_tokens, "seconds": round(d_seconds, 4), "tps": _rate(d_tokens, d_seconds)}
        record = {
            "method": self.method,
            "engine_type": self.engine_type,
            "model_type": self.model_type,
            "finish_reason": self.finish,
            "aborted": bool(aborted and self.finish is None),
            "prompt_tokens": self.prompt,
            "cached_tokens": min(self.cached, self.prompt) if self.prompt else self.cached,
            "cache_detail": self.detail,
            "completion_tokens": self.completion,
            "ttft_s": round(ttft, 4) if ttft is not None and ttft >= 0 else None,
            "prefill": prefill,
            "cached_prompt_tps": (
                _rate(self.prompt, ttft) if self.cached > 0 and ttft and ttft > 0 else None
            ),
            "decode": decode,
            "speculative": dict(self.spec) if self.spec is not None else None,
            "finished_at": time.time(),
        }
        if isinstance(self.request_id, str) and self.request_id:
            record["request_id"] = self.request_id
        global _LAST, _SEQ
        with _LOCK:
            _SEQ += 1
            record["seq"] = _SEQ
            _LAST = record


def mtp_spec_stats(record: Any) -> Optional[dict]:
    """Project native-MTP counters, preserving their reported measurement scope.

    A batched MLLM record describes its last MTP phase; request-wide finish and
    decode timing remain in the parent last_request record.
    """
    if not isinstance(record, dict):
        return None
    projected = {
        "kind": "mtp",
        "cycles": record.get("cycles"),
        "drafted": record.get("drafted_tokens"),
        "accepted": record.get("accepted_tokens"),
        "acceptance": record.get("acceptance_rate"),
        "final_depth": record.get("final_depth"),
        "depth_acceptance": record.get("depth_acceptance_rates"),
        "finish_reason": record.get("finish_reason"),
        "fallback_reason": record.get("fallback_reason"),
    }
    if record.get("scope"):
        projected["scope"] = record["scope"]
    return projected


def last() -> Optional[dict]:
    with _LOCK:
        return dict(_LAST) if _LAST is not None else None


def _seq() -> int:
    with _LOCK:
        return _SEQ


def reset() -> None:
    global _LAST, _SEQ
    with _LOCK:
        _LAST, _SEQ = None, 0


def instrument(engine: Any) -> Any:
    """Wrap ``stream_chat`` / ``stream_generate`` / ``chat`` / ``generate`` on this engine instance (idempotent)."""
    if engine is None or getattr(engine, "_vmlx_request_telemetry", False):
        return engine
    engine_type = {"BatchedEngine": "batched", "SimpleEngine": "simple"}.get(type(engine).__name__,
                                                                            type(engine).__name__)

    def model_type() -> str:
        try:
            return "mllm" if getattr(engine, "is_mllm", False) else "llm"
        except Exception:
            return "unknown"

    for name in ("stream_chat", "stream_generate"):
        original = getattr(engine, name, None)
        if original is None:
            continue

        def make_stream(original=original, name=name):
            @functools.wraps(original)
            async def tracked(*args, **kwargs):
                tracker = _Tracker(name, engine_type, model_type(), kwargs.get("request_id"))
                inner = original(*args, **kwargs)
                finished = False
                try:
                    async for output in inner:
                        tracker.observe(output)
                        yield output
                    finished = True
                finally:
                    tracker.record(aborted=not finished)
                    # Propagate an early close (client disconnect) to the engine's generator NOW: its own finally
                    # aborts the request; leaving that to garbage collection would keep decoding for nobody.
                    await inner.aclose()
            return tracked

        setattr(engine, name, make_stream())

    for name in ("chat", "generate"):
        original = getattr(engine, name, None)
        if original is None:
            continue

        def make_call(original=original, name=name):
            @functools.wraps(original)
            async def tracked(*args, **kwargs):
                before = _seq()
                tracker = _Tracker(name, engine_type, model_type(), kwargs.get("request_id"))
                output = await original(*args, **kwargs)
                # SimpleEngine.chat drains its own (instrumented) stream_chat: that record has the timings.
                if _seq() == before and output is not None:
                    tracker.observe(output)
                    tracker.record()
                return output
            return tracked

        setattr(engine, name, make_call())

    engine._vmlx_request_telemetry = True
    return engine

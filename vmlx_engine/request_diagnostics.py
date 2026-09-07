"""Per-request diagnostics that reach the response ``warnings`` array.

Two carriers, one sink:

* a context variable bucket, started by the HTTP handler before the engine
  runs (``begin_capture``) and drained when the response's ``warnings`` are
  assembled (``take``) — visible to code running in the request's task
  (tool-call filtering, the engine's message rewriting);
* a bounded registry keyed by request id (``record_for``) for code that runs
  OFF the request's task — the batch generator on the scheduler's executor
  thread has no view of the context variable. The engine drains the registry
  into the bucket right after generation (``drain_into_context``), so the
  same ``take`` delivers both.

Diagnostics are human-readable strings with a stable prefix per owner
(``video_controls:`` for effective video settings). They report what the
engine DID when it could not do exactly what was asked; they never replace
an error for a request that was refused.
"""

from __future__ import annotations

import contextvars
import threading
from collections import OrderedDict

DIAGNOSTICS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "vmlx_request_diagnostics", default=None
)

_REGISTRY: "OrderedDict[str, list[str]]" = OrderedDict()
_REGISTRY_LOCK = threading.Lock()
# Abandoned requests (client gone before the engine drained) must not grow
# the registry without bound; the oldest entries are dropped past this.
_REGISTRY_MAX_REQUESTS = 512


def begin_capture() -> None:
    """Start a fresh per-request bucket in the current context."""

    DIAGNOSTICS.set([])


def record(diagnostic: str) -> bool:
    """Append to the current request's bucket; False when no capture runs."""

    bucket = DIAGNOSTICS.get()
    if bucket is None:
        return False
    bucket.append(str(diagnostic))
    return True


def take() -> list[str]:
    """Return the accumulated diagnostics and reset the bucket."""

    bucket = DIAGNOSTICS.get()
    if not bucket:
        return []
    DIAGNOSTICS.set([])
    return list(bucket)


def record_for(request_id: str | None, diagnostic: str) -> bool:
    """Record for a request by id from ANY thread (drained by the engine)."""

    if not request_id:
        return False
    with _REGISTRY_LOCK:
        entries = _REGISTRY.get(request_id)
        if entries is None:
            entries = []
            _REGISTRY[request_id] = entries
            while len(_REGISTRY) > _REGISTRY_MAX_REQUESTS:
                _REGISTRY.popitem(last=False)
        entries.append(str(diagnostic))
    return True


def pending_for(request_id: str | None) -> list[str]:
    """Peek at a request's registry entries without draining them."""

    if not request_id:
        return []
    with _REGISTRY_LOCK:
        return list(_REGISTRY.get(request_id) or [])


def drain_into_context(request_id: str | None) -> int:
    """Move a request's registry entries into the current context bucket.

    Returns how many were delivered. Entries are dropped (not delivered) when
    no capture is running in this context, so a stale registry never leaks a
    diagnostic into a later, unrelated request that reuses nothing but the
    process.
    """

    if not request_id:
        return 0
    with _REGISTRY_LOCK:
        entries = _REGISTRY.pop(request_id, None)
    if not entries:
        return 0
    delivered = 0
    for text in entries:
        if record(text):
            delivered += 1
    return delivered

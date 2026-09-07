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
# The id the HTTP handler hands the engine for this request. ``take`` drains
# the registry for it, so entries recorded off-task (or before the handler
# opened its bucket) still reach the response that owns them.
CURRENT_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vmlx_request_diagnostics_id", default=None
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


def note_request_id(request_id: str | None) -> None:
    """Remember the engine-facing request id for the current context."""

    CURRENT_REQUEST_ID.set(str(request_id) if request_id else None)


def take() -> list[str]:
    """Return the accumulated diagnostics and reset the bucket.

    Entries recorded for the current request id by code outside this context
    (scheduler thread, or the engine before the handler opened its bucket)
    are drained here first, so the response that owns them delivers them.
    """

    request_id = CURRENT_REQUEST_ID.get()
    pending: list[str] = []
    if request_id:
        with _REGISTRY_LOCK:
            pending = _REGISTRY.pop(request_id, None) or []
    bucket = DIAGNOSTICS.get()
    if not bucket and not pending:
        return []
    DIAGNOSTICS.set([])
    return list(bucket or []) + pending


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

    Returns how many were delivered. When no capture is running in this
    context the entries stay in the registry for ``take`` (which drains by the
    handler's request id); the bounded registry drops the oldest requests, so
    an abandoned request never grows it without limit.
    """

    if not request_id or DIAGNOSTICS.get() is None:
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

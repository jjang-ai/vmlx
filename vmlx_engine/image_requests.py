"""Request-owned cancellation for the serialized image worker."""
import asyncio
import contextvars
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ImageRequestCancelled(RuntimeError):
    pass

@dataclass
class ImageRequest:
    request_id: str
    cancelled: threading.Event = field(default_factory=threading.Event)
    active: bool = False
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    def stop(self):
        self.cancelled.set()
        self.stopped.set()

    def check(self):
        if self.cancelled.is_set():
            raise ImageRequestCancelled(self.request_id)

current_image_request = contextvars.ContextVar("image_request", default=None)
_requests: dict[str, ImageRequest] = {}

def cancel_image_request(request_id=None):
    # Legacy no-id cancellation targets only the presently active request.
    # An idle cancel must never poison the next request.
    target = _requests.get(request_id) if request_id is not None else next(
        (r for r in _requests.values() if r.active), None)
    if target is None:
        return {"cancelled": False, "request_id": request_id}
    target.stop()
    logger.info("IMAGE_REQUEST request_id=%s phase=cancel_requested active=%s", target.request_id, target.active)
    return {"cancelled": True, "request_id": target.request_id, "state": "cancelling"}

@asynccontextmanager
async def image_request_scope(request, body, lock):
    request_id = body.get("request_id", request.headers.get("x-image-request-id"))
    if request_id is None:
        request_id = "imreq_" + uuid.uuid4().hex
    if not isinstance(request_id, str) or not request_id.strip() or not request_id.isprintable() or len(request_id) > 128:
        raise HTTPException(400, detail="request_id must be a nonempty printable string of at most 128 characters")
    if request_id in _requests:
        raise HTTPException(409, detail={"code": "image_request_id_in_use", "request_id": request_id})
    state = ImageRequest(request_id)
    _requests[request_id] = state
    logger.info("IMAGE_REQUEST request_id=%s phase=queued", request_id)
    async def watch():
        while True:
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                state.stop()
                logger.info("IMAGE_REQUEST request_id=%s phase=client_disconnected", request_id)
                return
    watcher = asyncio.create_task(watch())
    loop = asyncio.get_running_loop()
    acquire = loop.create_task(lock.acquire())
    cancelled = loop.create_task(state.stopped.wait())
    try:
        await asyncio.wait((acquire, cancelled), return_when=asyncio.FIRST_COMPLETED)
        state.check()
        await acquire
        state.active = True
        context_token = current_image_request.set(state)
        logger.info("IMAGE_REQUEST request_id=%s phase=active", request_id)
        try:
            yield state
            state.check()
        finally:
            current_image_request.reset(context_token)
            state.active = False
    except ImageRequestCancelled:
        raise HTTPException(409, detail={"code": "image_generation_cancelled", "request_id": request_id})
    finally:
        if acquire.done() and not acquire.cancelled() and acquire.exception() is None and acquire.result():
            lock.release()
        else:
            acquire.cancel()
        cancelled.cancel()
        await asyncio.gather(acquire, cancelled, return_exceptions=True)
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        _requests.pop(request_id, None)
        logger.info("IMAGE_REQUEST request_id=%s phase=finished cancelled=%s", request_id, state.cancelled.is_set())

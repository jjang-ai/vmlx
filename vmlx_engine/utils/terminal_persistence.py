# SPDX-License-Identifier: Apache-2.0
"""Terminal publication and request-owned cache persistence.

Default ("early"): a finished output is published the moment generation ends
and cache persistence settles in parallel. The next request's admission is
the only fence, so it still looks up a fully-published cache, while the
client can execute a tool or render the final answer without waiting for the
save. ``VMLX_TERMINAL_PUBLISH=strict`` restores the old behavior of holding
the terminal until persistence settles.
"""

import asyncio
import logging
import os
import time
from dataclasses import replace

from ..persistence_outcome import format_outcome

logger = logging.getLogger(__name__)

# Admission waits shorter than this are the normal open-gate path, not a
# prior save still in progress.
_ADMISSION_WAIT_LOG_S = 0.001


def strict_terminal_publish() -> bool:
    """True when terminals must wait for their cache persistence."""
    return os.environ.get("VMLX_TERMINAL_PUBLISH", "").strip().lower() == "strict"


def log_terminal_persistence_settled(request_id, persist_ms, pending_at_dispatch, outcome):
    logger.info(
        "Terminal persistence settled: request=%s persist_ms=%.3f "
        "pending_at_dispatch=%s published_early=%s %s",
        request_id,
        persist_ms,
        str(bool(pending_at_dispatch)).lower(),
        str(not strict_terminal_publish()).lower(),
        format_outcome(outcome),
    )


def log_admission_wait(request_id, wait_s):
    """Report a next request that waited for a prior terminal's persistence."""
    if wait_s >= _ADMISSION_WAIT_LOG_S:
        logger.info(
            "Admission waited for prior terminal persistence: request=%s wait_ms=%.3f",
            request_id,
            wait_s * 1000.0,
        )


def with_admission_wait(prefill_usage, wait_s):
    """Copy engine prefill usage with the admission wait reported beside it.

    The prefill seconds are measured from prefill start, after admission, so
    pp/s never absorbs the wait. Reporting it separately keeps a raw-API TTFT
    that includes leftover persistence attributable instead of looking like
    slow prefill.
    """
    if not isinstance(prefill_usage, dict) or wait_s is None:
        return prefill_usage
    usage = dict(prefill_usage)
    usage["admission_wait_seconds"] = round(float(wait_s), 6)
    return usage


async def settle_terminal_persistence(receipt, request_id, on_settled, dispatched_at):
    """Await a writer receipt after its terminal was already published.

    The writer owns the receipt; cancelling this waiter must not cancel the
    write. A settled failure is reported as failure, never as durable storage.
    """
    pending_at_dispatch = not receipt.done()
    outcome = await asyncio.shield(asyncio.wrap_future(receipt))
    persist_ms = (time.perf_counter() - dispatched_at) * 1000.0
    record = dict(
        outcome,
        request_id=request_id,
        wait_ms=persist_ms,
        waited=pending_at_dispatch,
        published_early=True,
    )
    on_settled(record)
    log_terminal_persistence_settled(request_id, persist_ms, pending_at_dispatch, outcome)
    return record


async def terminal_persistence_outputs(output, receipt, request_id, on_settled):
    """Strict mode: publish content promptly, but settle the write before EOF.

    The writer owns the receipt. Cancelling a reader must not cancel the write.
    A settled failure is reported as failure, never as durable storage.
    """
    produced_at = time.perf_counter()
    output = replace(output, generated_at=produced_at)
    if output.new_text:
        yield replace(output, finished=False, finish_reason=None)
        output = replace(output, new_text="")
    started = time.perf_counter()
    waited = not receipt.done()
    outcome = await asyncio.shield(asyncio.wrap_future(receipt))
    wait_ms = (time.perf_counter() - started) * 1000
    record = dict(outcome, request_id=request_id, wait_ms=wait_ms, waited=waited)
    on_settled(record)
    logger.info("Terminal durability barrier: request=%s wait_ms=%.3f waited=%s %s",
                request_id, wait_ms, str(waited).lower(), format_outcome(outcome))
    yield output

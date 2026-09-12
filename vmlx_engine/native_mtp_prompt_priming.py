# SPDX-License-Identifier: Apache-2.0
"""Prompt-history priming for native Qwen speculative decoding.

The target model already computes the trunk hidden state for every prompt
token.  Native MTP previously discarded those rows and began every request
with an empty head cache, so its first draft cycles were context-starved.  This
module folds the already-computed prompt hiddens into the MTP head cache and
keeps one exact, memory-only snapshot beside a matching full prefix-cache
block.  Every offset and hash boundary is validated; a mismatch falls back to
the existing unprimed path rather than guessing at history.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx


logger = logging.getLogger(__name__)

_CTX_ATTR = "_vmlx_native_mtp_prime_ctx"
_PLAN_ATTR = "_vmlx_native_mtp_prime_plan"
_LAST_ATTR = "_vmlx_native_mtp_prime_last"
_PARK_FOLD_BLOCK = 32


def _record(host: Any, reason: str, **detail: Any) -> None:
    setattr(host, _LAST_ATTR, {"reason": reason, **detail})


def priming_enabled() -> bool:
    return os.environ.get("VMLX_NATIVE_MTP_PROMPT_PRIMING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def prime_window() -> int:
    """Maximum newly folded prompt tokens; zero means no explicit cap."""
    try:
        return max(0, int(os.environ.get("VMLX_NATIVE_MTP_PRIME_WINDOW", "0")))
    except (TypeError, ValueError):
        return 0


@dataclass
class NativeMTPPrefixSnapshot:
    boundary_tokens: int
    mtp_cache: list[Any]
    pending_hidden: Any


@dataclass
class _BoundaryCandidate:
    boundary_tokens: int
    pending_hidden: Any


@dataclass
class _PrimePlan:
    request_id: str
    prompt_tokens: tuple[int, ...]
    block_size: int
    prefix_cache: Any
    extra_keys: Optional[tuple[Any, ...]] = None
    extra_key_token_start: Optional[int] = None
    extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None


@dataclass
class _PrimeContext:
    mtp_cache: list[Any]
    pending_hidden: Any = None
    folded: int = 0
    folded_this_request: int = 0
    expected_offset: int = 0
    request_id: Optional[str] = None
    prompt_tokens: Optional[tuple[int, ...]] = None
    block_size: int = 0
    prefix_cache: Any = None
    extra_keys: Optional[tuple[Any, ...]] = None
    extra_key_token_start: Optional[int] = None
    extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None
    boundary_candidate: Optional[_BoundaryCandidate] = None
    window_exceeded: bool = False
    # Request-local AR parking, never a prefix-cache/SSD snapshot. Keep the
    # already aligned head and at most one small block of confirmed pairs.
    parked: bool = False
    backbone_entries: tuple[Any, ...] = ()
    pending_pairs: list[tuple[Any, Any]] = field(default_factory=list)


def _eligible(host: Any) -> bool:
    return bool(
        host is not None
        and getattr(host, "mtp", None) is not None
        and callable(getattr(host, "mtp_forward", None))
        and callable(getattr(host, "make_mtp_cache", None))
    )


def drop_context(host: Any) -> None:
    if host is None:
        return
    for attr in (_CTX_ATTR, _PLAN_ATTR):
        if hasattr(host, attr):
            try:
                delattr(host, attr)
            except AttributeError:
                pass


def parked_priming_enabled() -> bool:
    """Experimental warm AR re-entry; no change to the default AR path."""
    return priming_enabled() and os.environ.get(
        "VMLX_NATIVE_MTP_PARKED_PRIMING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def parked_context_active(host: Any) -> bool:
    ctx = getattr(host, _CTX_ATTR, None)
    return isinstance(ctx, _PrimeContext) and ctx.parked


def drop_parked_context(host: Any, request_id: Optional[str] = None) -> None:
    """Release only the parked generation owned by this lifecycle event."""
    ctx = getattr(host, _CTX_ATTR, None)
    if not isinstance(ctx, _PrimeContext) or not ctx.parked:
        return
    if request_id is None or ctx.request_id == request_id:
        drop_context(host)


def _backbone_entries(cache: Any) -> tuple[Any, ...]:
    # The MLLM lane may create a fresh scalar-offset proxy for the same
    # BatchKVCache on every step. Compare the underlying objects, not offsets
    # alone: another request can have exactly the same context length.
    return tuple(getattr(entry, "_inner", entry) for entry in (cache or ()))


def _same_parked_backbone(ctx: _PrimeContext, cache: Any) -> bool:
    entries = _backbone_entries(cache)
    return len(entries) == len(ctx.backbone_entries) and all(
        actual is expected for actual, expected in zip(entries, ctx.backbone_entries)
    )


def capture_requested(host: Any) -> bool:
    """Return whether the scheduler armed prompt-history capture on ``host``.

    Patched upstream model wrappers use this to bypass their hidden-state-free
    fast path only for the one prompt currently being primed.  Keeping the
    check here avoids making every ordinary Qwen3.5 forward pay for the
    replacement wrapper merely because native MTP support is installed.
    """
    return isinstance(getattr(host, _PLAN_ATTR, None), _PrimePlan) or isinstance(
        getattr(host, _CTX_ATTR, None), _PrimeContext
    )


def _read_offset(entry: Any) -> Optional[int]:
    offset = getattr(entry, "offset", None)
    if type(offset) is int:
        return offset
    if offset is not None and getattr(offset, "size", 0) == 1:
        try:
            return int(offset.reshape(()).item())
        except Exception:
            return None
    return None


def _cache_offset(cache: Optional[list[Any]]) -> Optional[int]:
    for entry in cache or ():
        value = _read_offset(entry)
        if value is not None:
            return value
        for child in getattr(entry, "caches", ()) or ():
            value = _read_offset(child)
            if value is not None:
                return value
    return None


def _clone_cache(cache: list[Any]) -> list[Any]:
    def clone_entry(entry: Any) -> Any:
        children = getattr(entry, "caches", None)
        if children is not None:
            return type(entry)(*[clone_entry(child) for child in children])
        clone = copy.copy(entry)
        for attr, value in vars(entry).items():
            if isinstance(value, mx.array):
                setattr(clone, attr, value + 0)
            elif isinstance(value, list):
                setattr(
                    clone,
                    attr,
                    [
                        item + 0 if isinstance(item, mx.array) else item
                        for item in value
                    ],
                )
        return clone

    return [clone_entry(entry) for entry in cache]


def _flat_entries(cache: list[Any]):
    for entry in cache:
        children = getattr(entry, "caches", None)
        if children is None:
            yield entry
        else:
            yield from children


def _cache_at_offset(cache: list[Any], target: int) -> Optional[list[Any]]:
    if target < 0 or not cache:
        return None
    cloned = _clone_cache(cache)
    observed = False
    for entry in _flat_entries(cloned):
        current = _read_offset(entry)
        if current is None:
            continue
        observed = True
        if current < target:
            return None
        excess = current - target
        if excess:
            trim = getattr(entry, "trim", None)
            if not callable(trim) or int(trim(excess)) != excess:
                return None
        if _read_offset(entry) != target:
            return None
    return cloned if observed else None


def _snapshot_arrays(snapshot: NativeMTPPrefixSnapshot) -> list[Any]:
    arrays: list[Any] = []
    if isinstance(snapshot.pending_hidden, mx.array):
        arrays.append(snapshot.pending_hidden)
    for entry in _flat_entries(snapshot.mtp_cache):
        for value in vars(entry).values():
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, (list, tuple)):
                arrays.extend(item for item in value if isinstance(item, mx.array))
    return arrays


def _head_at_offset(cache: list[Any], offset: int) -> bool:
    entries = list(_flat_entries(cache))
    return bool(entries) and all(_read_offset(entry) == offset for entry in entries)


def park_after_confirmed(
    host: Any,
    *,
    request_id: str,
    backbone_cache: Any,
    mtp_cache: list[Any],
    confirmed_hidden: Any,
    confirmed_tokens: Any,
) -> bool:
    """Commit verified pairs and transfer their aligned head to AR parking.

    The caller must trim speculative head-chain rows first. At this seam the
    last queued token has not entered the backbone yet, but its preceding
    hidden/token pair belongs in the head. The first ordinary AR forward
    captures that token's hidden without folding the same pair twice.
    """
    if not parked_priming_enabled() or not _eligible(host) or not mtp_cache:
        return False
    offset = _cache_offset(backbone_cache)
    if (
        offset is None
        or getattr(confirmed_tokens, "ndim", 0) != 2
        or confirmed_tokens.shape[0] != 1
        or confirmed_hidden.shape[:2] != confirmed_tokens.shape
    ):
        return False
    count = int(confirmed_tokens.shape[1])
    if count <= 0 or not _head_at_offset(mtp_cache, offset - count):
        _record(host, "park_head_offset_mismatch", expected_offset=offset)
        return False
    try:
        # No proposal sampling or logits evaluation: only confirmed cache
        # state is needed. Target/verifier cache objects are never written.
        host.mtp_forward(confirmed_hidden, confirmed_tokens, mtp_cache)
        if not _head_at_offset(mtp_cache, offset):
            _record(host, "park_commit_offset_mismatch", expected_offset=offset)
            return False
        ctx = _PrimeContext(
            mtp_cache=mtp_cache,
            folded=offset,
            expected_offset=offset,
            request_id=request_id,
            parked=True,
            backbone_entries=_backbone_entries(backbone_cache),
        )
        drop_context(host)
        setattr(host, _CTX_ATTR, ctx)
        _record(host, "parked", folded_pairs=offset, request_id=request_id)
        logger.info(
            "MLLM MTP[%s] parked aligned head at %d pairs (buffer cap=%d)",
            request_id, offset, _PARK_FOLD_BLOCK,
        )
        return True
    except Exception:
        drop_parked_context(host, request_id)
        logger.debug("native MTP park commit failed closed", exc_info=True)
        return False


def _flush_parked(host: Any, ctx: _PrimeContext) -> None:
    if not ctx.pending_pairs:
        return
    hidden = mx.concatenate([pair[0] for pair in ctx.pending_pairs], axis=1)
    tokens = mx.concatenate([pair[1] for pair in ctx.pending_pairs], axis=1)
    host.mtp_forward(hidden, tokens, ctx.mtp_cache)
    ctx.folded += int(tokens.shape[1])
    ctx.pending_pairs.clear()
    if not _head_at_offset(ctx.mtp_cache, ctx.folded):
        raise ValueError("parked MTP fold did not advance every head layer")
    arrays = _snapshot_arrays(
        NativeMTPPrefixSnapshot(ctx.folded, ctx.mtp_cache, ctx.pending_hidden)
    )
    if arrays:
        mx.async_eval(*arrays)


def _capture_parked(
    host: Any, ctx: _PrimeContext, inputs: Any, hidden: Any, offset_after: int
) -> None:
    if inputs.shape[1] != 1 or hidden.shape[:2] != inputs.shape:
        _record(host, "park_non_singleton_forward")
        drop_context(host)
        return
    if ctx.pending_hidden is not None:
        ctx.pending_pairs.append((ctx.pending_hidden, inputs))
    ctx.pending_hidden = hidden[:, -1:]
    ctx.expected_offset = offset_after
    try:
        if len(ctx.pending_pairs) >= _PARK_FOLD_BLOCK:
            _flush_parked(host, ctx)
    except Exception:
        _record(host, "park_fold_failed")
        drop_context(host)
        logger.debug("native MTP parked fold failed closed", exc_info=True)


def prepare_prompt(
    host: Any,
    *,
    request_id: str,
    prompt_tokens: list[int],
    cached_tokens: int,
    prefix_cache: Any,
    extra_keys: Optional[tuple[Any, ...]] = None,
    extra_key_token_start: Optional[int] = None,
    extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None,
) -> bool:
    """Arm one exact prompt timeline and restore its MTP sidecar if present."""
    if not priming_enabled() or not _eligible(host):
        _record(
            host,
            "prepare_disabled" if not priming_enabled() else "prepare_ineligible",
        )
        drop_context(host)
        return False

    tokens = tuple(int(token) for token in prompt_tokens)
    cached_tokens = max(0, int(cached_tokens))
    drop_context(host)
    plan = _PrimePlan(
        request_id=request_id,
        prompt_tokens=tokens,
        block_size=max(0, int(getattr(prefix_cache, "block_size", 0) or 0)),
        prefix_cache=prefix_cache,
        extra_keys=extra_keys,
        extra_key_token_start=extra_key_token_start,
        extra_key_ranges=(
            list(extra_key_ranges) if extra_key_ranges is not None else None
        ),
    )
    setattr(host, _PLAN_ATTR, plan)
    _record(
        host,
        "armed",
        prompt_tokens=len(tokens),
        cached_tokens=cached_tokens,
    )
    if cached_tokens <= 0:
        return False

    restore = getattr(prefix_cache, "restore_mtp_prefix_snapshot", None)
    if not callable(restore):
        return False
    try:
        snapshot = restore(
            list(tokens),
            cached_tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
    except Exception:
        logger.debug("native MTP sidecar lookup failed closed", exc_info=True)
        return False
    if not isinstance(snapshot, NativeMTPPrefixSnapshot):
        return False
    if snapshot.boundary_tokens != cached_tokens or cached_tokens < 2:
        return False
    restored = _cache_at_offset(snapshot.mtp_cache, cached_tokens - 1)
    if restored is None or snapshot.pending_hidden is None:
        return False
    pending = snapshot.pending_hidden + 0
    setattr(
        host,
        _CTX_ATTR,
        _PrimeContext(
            mtp_cache=restored,
            pending_hidden=pending,
            folded=cached_tokens - 1,
            expected_offset=cached_tokens,
            request_id=request_id,
            prompt_tokens=tokens,
            block_size=plan.block_size,
            prefix_cache=prefix_cache,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=plan.extra_key_ranges,
        ),
    )
    mx.async_eval(pending)
    logger.info(
        "MLLM native MTP prompt history restored at %d tokens for %s",
        cached_tokens,
        request_id,
    )
    return True


def _capture_boundary(ctx: _PrimeContext, hidden: Any, start: int, end: int) -> None:
    block = int(ctx.block_size or 0)
    if block <= 0 or ctx.prefix_cache is None or not ctx.prompt_tokens:
        return
    full_block_boundary = (end // block) * block
    terminal_boundary = len(ctx.prompt_tokens) - 1
    boundary = full_block_boundary
    if start < terminal_boundary <= end:
        # vMLX stores the predecessor's exact N-1 terminal cache and can
        # restore that non-block-aligned boundary on an identical request.
        boundary = max(boundary, terminal_boundary)
    # The backbone prefix cache never holds more than the N-1 terminal
    # boundary of this prompt, so a sidecar keyed past it can never be
    # restored.  Measured 2026-09-04 (Flash-Next 4S, 14-turn chat): a prompt
    # of exactly 5952 tokens (93 x 64) published its sidecar at 5952 while
    # every later turn restored the prefix at 5951; priming was lost for the
    # rest of the conversation and MTP ran unprimed (acceptance 77-84% ->
    # 44-63%, decode 45-60 -> 25-35 tok/s).
    boundary = min(boundary, terminal_boundary)
    if boundary <= start or boundary > len(ctx.prompt_tokens) or boundary <= 1:
        return
    prior = ctx.boundary_candidate
    if prior is not None and prior.boundary_tokens >= boundary:
        return
    if ctx.folded < boundary - 1:
        return
    row = boundary - start - 1
    if row < 0 or row >= int(hidden.shape[1]):
        return
    candidate_hidden = hidden[:, row : row + 1] + 0
    mx.async_eval(candidate_hidden)
    ctx.boundary_candidate = _BoundaryCandidate(boundary, candidate_hidden)


def capture_prefill(host: Any, inputs: Any, expanded_hidden: Any, cache: Any) -> None:
    """Fold one contiguous Qwen prompt forward into the native MTP cache."""
    if not priming_enabled():
        drop_parked_context(host)
        _record(host, "capture_disabled")
        return
    if not _eligible(host):
        _record(host, "capture_ineligible")
        return
    if inputs is None or getattr(inputs, "ndim", 0) != 2 or inputs.shape[0] != 1:
        _record(host, "capture_invalid_inputs")
        drop_context(host)
        return
    plan = getattr(host, _PLAN_ATTR, None)
    ctx = getattr(host, _CTX_ATTR, None)
    if not isinstance(plan, _PrimePlan) and not isinstance(ctx, _PrimeContext):
        return
    if isinstance(ctx, _PrimeContext) and ctx.parked and (
        not parked_priming_enabled() or not _same_parked_backbone(ctx, cache)
    ):
        _record(host, "park_backbone_changed_or_disabled")
        drop_context(host)
        return
    offset_after = _cache_offset(cache)
    if offset_after is None:
        _record(host, "capture_unknown_cache_offset")
        drop_context(host)
        return
    seq_len = int(inputs.shape[1])
    start = offset_after - seq_len
    if isinstance(ctx, _PrimeContext) and ctx.expected_offset != start:
        _record(
            host,
            "capture_noncontiguous_offset",
            expected_offset=ctx.expected_offset,
            actual_start=start,
        )
        drop_context(host)
        return
    if isinstance(ctx, _PrimeContext) and ctx.parked:
        _capture_parked(host, ctx, inputs, expanded_hidden, offset_after)
        return
    if isinstance(ctx, _PrimeContext) and ctx.window_exceeded:
        _record(host, "capture_window_already_exceeded")
        ctx.expected_offset = offset_after
        return
    window = prime_window()
    folded_this_request = ctx.folded_this_request if isinstance(ctx, _PrimeContext) else 0
    if window and folded_this_request + seq_len > window:
        _record(
            host,
            "capture_window_exceeded",
            window=window,
            attempted_pairs=folded_this_request + seq_len,
        )
        setattr(
            host,
            _CTX_ATTR,
            _PrimeContext(
                mtp_cache=[],
                expected_offset=offset_after,
                window_exceeded=True,
            ),
        )
        return
    if not isinstance(ctx, _PrimeContext):
        # A restored backbone prefix without its exact MTP sidecar cannot be
        # represented by starting a fresh head cache at the uncached tail: its
        # QSA positions and attention history would begin at zero.  Stay on
        # the existing unprimed activation path instead.
        if start != 0:
            _record(host, "capture_nonzero_start", actual_start=start)
            return
        if seq_len <= 1:
            _record(host, "capture_single_token_start")
            return
        assert isinstance(plan, _PrimePlan)
        mtp_cache = host.make_mtp_cache()
        if not mtp_cache:
            _record(host, "capture_empty_mtp_cache")
            return
        ctx = _PrimeContext(
            mtp_cache=mtp_cache,
            request_id=plan.request_id,
            prompt_tokens=plan.prompt_tokens,
            block_size=plan.block_size,
            prefix_cache=plan.prefix_cache,
            extra_keys=plan.extra_keys,
            extra_key_token_start=plan.extra_key_token_start,
            extra_key_ranges=plan.extra_key_ranges,
        )
        setattr(host, _CTX_ATTR, ctx)

    if ctx.pending_hidden is None:
        if seq_len <= 1:
            ctx.pending_hidden = expanded_hidden[:, -1:]
            ctx.expected_offset = offset_after
            return
        pair_hidden = expanded_hidden[:, :-1]
        pair_tokens = inputs[:, 1:]
    else:
        pair_hidden = (
            mx.concatenate([ctx.pending_hidden, expanded_hidden[:, :-1]], axis=1)
            if seq_len > 1
            else ctx.pending_hidden
        )
        pair_tokens = inputs

    # The logits tail is intentionally left lazy and discarded.  The head
    # cache writes are materialized below, which is the only state we need.
    host.mtp_forward(pair_hidden, pair_tokens, ctx.mtp_cache)
    pairs = int(pair_tokens.shape[1])
    ctx.folded += pairs
    ctx.folded_this_request += pairs
    ctx.pending_hidden = expanded_hidden[:, -1:]
    ctx.expected_offset = offset_after
    _record(
        host,
        "captured",
        folded_pairs=ctx.folded,
        expected_offset=ctx.expected_offset,
    )
    _capture_boundary(ctx, expanded_hidden, start, offset_after)

    arrays = [ctx.pending_hidden]
    for entry in _flat_entries(ctx.mtp_cache):
        for value in vars(entry).values():
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, (list, tuple)):
                arrays.extend(item for item in value if isinstance(item, mx.array))
    mx.async_eval(*arrays)


def _publish_boundary(ctx: _PrimeContext) -> None:
    candidate = ctx.boundary_candidate
    store = getattr(ctx.prefix_cache, "store_mtp_prefix_snapshot", None)
    if candidate is None or not callable(store):
        return
    snapshot_cache = _cache_at_offset(
        ctx.mtp_cache, candidate.boundary_tokens - 1
    )
    if snapshot_cache is None:
        return
    snapshot = NativeMTPPrefixSnapshot(
        boundary_tokens=candidate.boundary_tokens,
        mtp_cache=snapshot_cache,
        pending_hidden=candidate.pending_hidden,
    )
    arrays = _snapshot_arrays(snapshot)
    if arrays:
        mx.async_eval(*arrays)
    try:
        stored = store(
            list(ctx.prompt_tokens or ()),
            candidate.boundary_tokens,
            snapshot,
            extra_keys=ctx.extra_keys,
            extra_key_token_start=ctx.extra_key_token_start,
            extra_key_ranges=ctx.extra_key_ranges,
        )
    except Exception:
        logger.debug("native MTP sidecar publish failed closed", exc_info=True)
        return
    if stored:
        logger.info(
            "MLLM native MTP prompt history cached at %d tokens for %s",
            candidate.boundary_tokens,
            ctx.request_id or "unknown",
        )


def take_primed(host: Any, backbone_cache: Any, main_token: Any) -> Optional[tuple[list[Any], int]]:
    """Consume a contiguous primed history at the first MTP draft seam."""
    ctx = getattr(host, _CTX_ATTR, None)
    if not isinstance(ctx, _PrimeContext):
        drop_context(host)
        return None
    drop_context(host)
    if ctx.window_exceeded or ctx.folded <= 0 or ctx.pending_hidden is None:
        return None
    offset = _cache_offset(backbone_cache)
    if (
        offset is None or ctx.expected_offset != offset - 1
        or (ctx.parked and (
            not parked_priming_enabled()
            or not _same_parked_backbone(ctx, backbone_cache)
        ))
    ):
        logger.info(
            "MLLM native MTP priming discarded at seam: expected=%s actual=%s",
            ctx.expected_offset,
            offset,
        )
        return None
    try:
        if ctx.parked:
            _flush_parked(host, ctx)
        host.mtp_forward(
            ctx.pending_hidden,
            main_token.reshape(1, 1),
            ctx.mtp_cache,
        )
        if ctx.parked and not _head_at_offset(ctx.mtp_cache, ctx.folded + 1):
            raise ValueError("parked MTP seam did not advance every head layer")
    except Exception:
        logger.debug("native MTP priming seam failed closed", exc_info=True)
        return None
    _publish_boundary(ctx)
    if ctx.parked:
        _record(host, "park_resumed", folded_pairs=ctx.folded + 1)
        logger.info(
            "MLLM MTP[%s] resumed parked head at %d pairs",
            ctx.request_id, ctx.folded + 1,
        )
    return ctx.mtp_cache, ctx.folded + 1


def prime_stats(host: Any) -> dict[str, Any]:
    ctx = getattr(host, _CTX_ATTR, None)
    plan = getattr(host, _PLAN_ATTR, None)
    return {
        "plan_armed": isinstance(plan, _PrimePlan),
        "active": isinstance(ctx, _PrimeContext) and not ctx.window_exceeded,
        "folded_pairs": int(ctx.folded) if isinstance(ctx, _PrimeContext) else 0,
        "window_exceeded": bool(
            isinstance(ctx, _PrimeContext) and ctx.window_exceeded
        ),
        "last": dict(getattr(host, _LAST_ATTR, {}) or {}),
    }


__all__ = [
    "NativeMTPPrefixSnapshot",
    "capture_prefill",
    "drop_context",
    "drop_parked_context",
    "park_after_confirmed",
    "parked_context_active",
    "parked_priming_enabled",
    "prepare_prompt",
    "prime_stats",
    "priming_enabled",
    "take_primed",
]

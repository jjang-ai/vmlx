# SPDX-License-Identifier: Apache-2.0
"""Hybrid SSM partial-accept replay helper — issue #134.

Factored out of Scheduler._replay_ssm_forward so tests can import and
exercise the production code path directly without pulling in the full
vmlx_engine.scheduler module.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def replay_ssm_forward(
    model,
    kv_cache: list,
    saved_array_caches: Dict[int, list],
    accepted_tokens: List[int],
    pre_verify_offset: int,
) -> bool:
    """Replay accepted_tokens through model to advance SSM+KV caches to N+K'.

    After hybrid partial rejection, restores both caches to N, then runs a
    single forward pass over accepted_tokens to reach N+num_accept. The logits
    are discarded; only the cache side-effect matters.

    Args:
        model: The language model callable (model(input, cache=...) -> logits).
        kv_cache: Per-layer cache list (mix of SSM ArraysCache + KVCache).
        saved_array_caches: Snapshot dict {layer_idx: list_of_arrays} captured
            before the verify forward pass.
        accepted_tokens: Draft tokens that were accepted (length = num_accept).
        pre_verify_offset: KV offset N before the verify forward ran.

    Returns:
        True on success (caches at N+num_accept).
        False on failure (caches restored to pre_verify_offset by except handler).
    """
    import mlx.core as mx
    import numpy as _np_local

    # Lazy import: generation_stream may not be available in minimal test envs.
    try:
        from mlx_lm.generate import generation_stream as _gen_stream
        _stream_ctx = mx.stream(_gen_stream)
    except Exception:
        _stream_ctx = contextlib.nullcontext()

    # Lazy import: CacheList for RotatingKVCache-based lists.
    try:
        from mlx_lm.models.cache import CacheList as _CL
    except ImportError:
        _CL = None

    def _rewind_kv_to(target_offset: int) -> None:
        for c in kv_cache:
            if not c.is_trimmable() or c.offset == 0:
                continue
            if c.offset <= target_offset:
                continue
            if _CL is not None and isinstance(c, _CL):
                c.trim(c.offset - target_offset)
                continue
            if isinstance(c.keys, mx.array):
                _kd, _vd = c.keys.dtype, c.values.dtype
                _ka = c.keys.astype(mx.float16) if "bfloat16" in str(_kd) else c.keys
                _va = c.values.astype(mx.float16) if "bfloat16" in str(_vd) else c.values
                _k, _v = _np_local.array(_ka), _np_local.array(_va)
                c.keys = mx.array(_k[..., :target_offset, :]).astype(_kd)
                c.values = mx.array(_v[..., :target_offset, :]).astype(_vd)
            c.offset = target_offset
            if hasattr(c, "_idx"):
                c._idx = target_offset

    try:
        # 1. Restore ArraysCache layers to pre-verify snapshot
        for i, c in enumerate(kv_cache):
            if i in saved_array_caches:
                c.cache = saved_array_caches[i]

        # 2. Rewind KV layers to pre_verify_offset
        _rewind_kv_to(pre_verify_offset)

        # 3. Replay forward: shape (1, num_accept) — advances caches to N+num_accept
        replay_input = mx.array([accepted_tokens])
        with _stream_ctx:
            _ = model(replay_input, cache=kv_cache)
            mx.eval(kv_cache)

        return True

    except Exception as exc:
        logger.warning("[PLD-replay] SSM replay failed: %s", exc, exc_info=False)
        # Best-effort restore: re-apply snapshot, re-rewind KV
        try:
            for i, c in enumerate(kv_cache):
                if i in saved_array_caches:
                    c.cache = saved_array_caches[i]
            _rewind_kv_to(pre_verify_offset)
        except Exception:
            pass
        return False

"""Cache-record validator: hard guard against malformed paged/L2 cache restores.

Codex 2026-05-06 contract item #4: validate BEFORE allocating MLX tensors.
A corrupted L2 disk block (stale schema, wrong model, mid-write tear) used to
cascade into ``cache.state = <bad-shape>`` then trigger
``[metal::malloc] Attempting to allocate 468462801024 bytes`` on the next forward
pass. This module rejects malformed records up-front so the scheduler treats them
as cache-miss and re-prefills cleanly.

Validator is intentionally cheap (operates on numpy/MLX shapes; no MLX kernel
launches) and idempotent — call from both ``block_disk_store.read_block`` and
``prefix_cache.reconstruct_cache``.

The byte caps below are SAFETY ceilings, not target values. A well-behaved DSV4
cache record fits well under them. Anything that *would* blow past these caps is
guaranteed-corrupt or guaranteed-OOM-on-this-machine, so cache-miss is strictly
better than allocate-and-die.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- Hard ceilings (Codex #4: "no decoded tensor may request hundreds of GB")
# Per-tensor cap: 4 GB. The largest legitimate single tensor in any vMLX cache
# record is the full SWA RotatingKVCache for a 64K-token window at f16, which
# tops out around 1-2 GB. 4 GB is a generous safety margin without giving up
# the ability to detect corruption (a typical corrupt record requests 100+ GB).
MAX_TENSOR_BYTES = 4 * 1024 ** 3

# Per-record cap: 16 GB. Sums all tensors in a single block's cache_data. Even
# a giant 43-layer DSV4 block at full quantization stays under 8 GB. 16 GB
# leaves headroom for future model families with more layers/heads.
MAX_TOTAL_RECORD_BYTES = 16 * 1024 ** 3

# Per-dimension sanity cap. A single tensor dim of 64K is plausible (max paged
# token count). A dim of 1M is corruption.
MAX_TENSOR_DIM = 65536 * 4  # 256K — defensive 4x

# Maximum tensor rank (ndim). KV caches are at most rank-4
# (batch, heads, seq, head_dim). State trees with >6 ndim are corruption.
MAX_TENSOR_NDIM = 6


class CacheValidationError(ValueError):
    """Raised by ``validate_cache_record`` when a record is unsafe to restore."""


def _tensor_byte_size(t: Any) -> int:
    """Compute byte size of an MLX/numpy tensor without triggering eval."""
    if t is None:
        return 0
    nbytes = getattr(t, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    shape = getattr(t, "shape", None)
    itemsize = getattr(t, "itemsize", None)
    if shape is None or itemsize is None:
        return 0
    n = 1
    for d in shape:
        n *= max(int(d), 0)
    return n * int(itemsize)


def _validate_tensor(
    t: Any,
    *,
    label: str,
    layer_idx: int,
) -> Tuple[bool, int, str]:
    """Validate a single tensor's shape + byte size. Returns (ok, bytes, reason)."""
    if t is None:
        return True, 0, ""
    shape = getattr(t, "shape", None)
    if shape is None:
        # Some entries are scalars or non-tensors (e.g. rotating-kv max_size).
        return True, 0, ""
    if len(shape) > MAX_TENSOR_NDIM:
        return False, 0, (
            f"layer {layer_idx} {label}: ndim={len(shape)} > {MAX_TENSOR_NDIM}"
        )
    for axis, dim in enumerate(shape):
        d = int(dim)
        if d < 0:
            return False, 0, f"layer {layer_idx} {label}: dim[{axis}]={d} < 0"
        if d > MAX_TENSOR_DIM:
            return False, 0, (
                f"layer {layer_idx} {label}: dim[{axis}]={d} > {MAX_TENSOR_DIM}"
            )
    nbytes = _tensor_byte_size(t)
    if nbytes > MAX_TENSOR_BYTES:
        return False, nbytes, (
            f"layer {layer_idx} {label}: {nbytes} bytes > {MAX_TENSOR_BYTES} cap"
        )
    return True, nbytes, ""


def _walk_tensors(obj: Any) -> Iterable[Any]:
    """Yield tensor-like objects nested inside dict/list/tuple state trees."""
    if obj is None:
        return
    if hasattr(obj, "shape"):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_tensors(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_tensors(v)


def validate_cache_record(
    cache_data: Any,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> Tuple[bool, str, int]:
    """Validate a cache_data list before tensor reconstruction.

    Args:
        cache_data: The list of per-layer entries returned by
            ``_deserialize_block`` or ``_extract_block_tensor_slice``. Each entry
            is a tuple whose first element is a tag string (``"kv"``,
            ``"quantized_kv"``, ``"rotating_kv"``, ``"cumulative"``,
            ``"deepseek_v4"``, ``"deepseek_v4_pending"``, ``"cache_list"``,
            ``"skip"``).
        expected_num_layers: If not None, ``len(cache_data)`` must match.
        source: Tag for logging (e.g. ``"L2-disk"``, ``"reconstruct"``).

    Returns:
        (ok, reason, total_bytes). ``ok=False`` means the caller MUST treat the
        record as cache-miss and never feed it to MLX.
    """
    if not isinstance(cache_data, list):
        return False, f"cache_data is {type(cache_data).__name__}, expected list", 0

    n = len(cache_data)
    if n == 0:
        return False, "cache_data is empty", 0

    if expected_num_layers is not None and n != expected_num_layers:
        # Layer-count mismatch is the canonical "wrong model / wrong schema"
        # signal. A DSV4-Flash 43-layer block in a Qwen 80-layer scheduler would
        # otherwise reconstruct silently and hit a downstream shape mismatch.
        return False, (
            f"layer count {n} != expected {expected_num_layers} (source={source})"
        ), 0

    total_bytes = 0
    for i, entry in enumerate(cache_data):
        if not isinstance(entry, tuple) or len(entry) < 1:
            return False, (
                f"layer {i}: malformed entry {type(entry).__name__} "
                f"len={len(entry) if hasattr(entry, '__len__') else 'n/a'}"
            ), total_bytes

        tag = entry[0]

        if tag in ("skip", "deepseek_v4_pending"):
            continue

        if tag == "kv":
            # ("kv", keys, values)
            if len(entry) < 3:
                return False, f"layer {i} 'kv': len={len(entry)} < 3", total_bytes
            for sub_label, t in (("keys", entry[1]), ("values", entry[2])):
                ok, nb, reason = _validate_tensor(t, label=sub_label, layer_idx=i)
                if not ok:
                    return False, reason, total_bytes
                total_bytes += nb

        elif tag == "rotating_kv":
            # ("rotating_kv", keys, values, max_size, keep)
            if len(entry) < 3:
                return False, (
                    f"layer {i} 'rotating_kv': len={len(entry)} < 3"
                ), total_bytes
            for sub_label, t in (("keys", entry[1]), ("values", entry[2])):
                ok, nb, reason = _validate_tensor(t, label=sub_label, layer_idx=i)
                if not ok:
                    return False, reason, total_bytes
                total_bytes += nb

        elif tag == "quantized_kv":
            # ("quantized_kv", (data, scales, zeros), (data, scales, zeros), meta?)
            if len(entry) < 3:
                return False, (
                    f"layer {i} 'quantized_kv': len={len(entry)} < 3"
                ), total_bytes
            for tup_label, tup in (("keys", entry[1]), ("values", entry[2])):
                if not isinstance(tup, (tuple, list)) or len(tup) != 3:
                    return False, (
                        f"layer {i} 'quantized_kv' {tup_label}: "
                        f"expected 3-tuple, got {type(tup).__name__} "
                        f"len={len(tup) if hasattr(tup, '__len__') else 'n/a'}"
                    ), total_bytes
                for j, sub in enumerate(tup):
                    ok, nb, reason = _validate_tensor(
                        sub, label=f"{tup_label}[{j}]", layer_idx=i
                    )
                    if not ok:
                        return False, reason, total_bytes
                    total_bytes += nb

        elif tag == "cumulative":
            # ("cumulative", state_arrays, meta, class_name)
            if len(entry) < 2:
                return False, (
                    f"layer {i} 'cumulative': len={len(entry)} < 2"
                ), total_bytes
            for j, t in enumerate(entry[1] or []):
                ok, nb, reason = _validate_tensor(
                    t, label=f"state[{j}]", layer_idx=i
                )
                if not ok:
                    return False, reason, total_bytes
                total_bytes += nb

        elif tag == "deepseek_v4":
            # ("deepseek_v4", state, meta, class_name, cache_meta)
            if len(entry) < 2:
                return False, (
                    f"layer {i} 'deepseek_v4': len={len(entry)} < 2"
                ), total_bytes
            # State is a nested tree (dict/list/tuple of tensors). Walk it.
            for j, t in enumerate(_walk_tensors(entry[1])):
                ok, nb, reason = _validate_tensor(
                    t, label=f"dsv4_state_t{j}", layer_idx=i
                )
                if not ok:
                    return False, reason, total_bytes
                total_bytes += nb

        elif tag == "cache_list":
            # ("cache_list", [sub_entries])
            if len(entry) < 2:
                return False, (
                    f"layer {i} 'cache_list': len={len(entry)} < 2"
                ), total_bytes
            sub_entries = entry[1] or []
            # Recurse: each sub_entry is itself a tagged tuple. Reuse the
            # validator on a synthetic single-layer record.
            sub_ok, sub_reason, sub_bytes = validate_cache_record(
                list(sub_entries),
                expected_num_layers=None,
                source=f"{source}/cache_list[{i}]",
            )
            if not sub_ok:
                return False, f"layer {i} cache_list: {sub_reason}", total_bytes
            total_bytes += sub_bytes

        else:
            # Unknown tag — treat as malformed rather than silently accept.
            return False, (
                f"layer {i}: unknown tag {tag!r} (source={source})"
            ), total_bytes

        if total_bytes > MAX_TOTAL_RECORD_BYTES:
            return False, (
                f"total_bytes {total_bytes} > {MAX_TOTAL_RECORD_BYTES} "
                f"(source={source})"
            ), total_bytes

    return True, "", total_bytes


def reject_or_warn(
    cache_data: Any,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> bool:
    """Validate + log. Returns True if record is safe, False if must be rejected.

    Caller pattern:
        if not reject_or_warn(cache_data, expected_num_layers=43, source="L2"):
            return None  # treat as cache miss
    """
    ok, reason, nbytes = validate_cache_record(
        cache_data,
        expected_num_layers=expected_num_layers,
        source=source,
    )
    if not ok:
        # WARNING (not ERROR): a stale L2 entry from an older schema is expected
        # behavior and recovers gracefully via cache-miss + re-prefill. Loud
        # enough that ops can spot a flood (indicating real corruption).
        logger.warning(
            "Cache validation rejected record from %s: %s "
            "(bytes_seen=%d). Treating as cache miss.",
            source, reason, nbytes,
        )
    return ok

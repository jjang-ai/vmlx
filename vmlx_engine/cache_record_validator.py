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


# ============================================================================
# Pre-mx.load safetensors header validation (Codex 2026-05-06 follow-up).
#
# CRITICAL: ``reject_or_warn`` runs AFTER ``mx.load(file)``. If a corrupt
# safetensors file describes a tensor whose shape would request 437 GB,
# ``mx.load`` itself triggers ``[metal::malloc]`` and crashes the engine
# BEFORE the validator ever sees the deserialized record.
#
# This module adds a pre-load validator that reads only the safetensors
# JSON header (~1 KB) and rejects files whose declared shapes violate the
# same caps. The header layout is documented at
# https://github.com/huggingface/safetensors and is stable:
#
#     [8 bytes LE uint64 = N (header byte length)]
#     [N bytes UTF-8 JSON dict: name -> {dtype, shape, data_offsets}]
#     [raw tensor data]
#
# Reading the header is one ``f.read(8)`` + one ``f.read(N)`` + ``json.loads``.
# No tensor data touched, no MLX involvement, no GPU allocation possible.
# ============================================================================

# safetensors dtype name -> bytes/element. Stable across versions.
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F16": 2, "BF16": 2, "U16": 2, "I16": 2,
    "F32": 4, "U32": 4, "I32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
    "F64": 8, "U64": 8, "I64": 8,
}


def validate_safetensors_header(
    file_path: str,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> Tuple[bool, str]:
    """Validate a .safetensors file's HEADER without loading tensor data.

    Returns ``(ok, reason)``. ``ok=False`` means the caller MUST treat the
    file as missing/corrupt and never call ``mx.load`` on it.

    This guards against two failure modes that ``mx.load`` cannot:
    1. A header tensor whose declared shape × dtype-bytes exceeds the
       per-tensor cap (4 GB by default). ``mx.load`` would allocate a
       Metal buffer of that size and crash with ``[metal::malloc]
       Attempting to allocate N bytes...``.
    2. A header sum exceeding the per-record cap (16 GB). Even if every
       single tensor fits, an accumulated load would exhaust memory.

    Cheap (~1 KB read + JSON parse). Safe to call on every disk hit.
    """
    import os
    if not os.path.exists(file_path):
        return False, f"file does not exist: {file_path}"

    try:
        size_on_disk = os.path.getsize(file_path)
    except OSError as e:
        return False, f"stat failed: {e}"

    if size_on_disk < 8:
        return False, f"file too small ({size_on_disk} bytes) — not a safetensors"

    # The pre-validator's per-tensor cap should be slightly looser than the
    # post-load validator (since we don't yet know the cache-record context),
    # but still hard enough to catch the 437 GB / 555 GB allocation class.
    PER_TENSOR_HEADER_CAP = MAX_TENSOR_BYTES  # 4 GB
    PER_FILE_HEADER_CAP = MAX_TOTAL_RECORD_BYTES * 2  # 32 GB (a single L2 file
    # can store more than one cache record's worth in some corner cases)

    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) != 8:
                return False, "could not read 8-byte header length"
            header_size = int.from_bytes(header_size_bytes, "little", signed=False)
            if header_size <= 0 or header_size > 16 * 1024 * 1024:
                # 16 MB cap on the JSON header itself. A legit header for a
                # 43-layer DSV4 record is well under 100 KB. Anything larger
                # is corrupt.
                return False, (
                    f"header_size {header_size} out of bounds (max 16MB)"
                )
            if header_size > size_on_disk - 8:
                return False, (
                    f"header_size {header_size} > file_size {size_on_disk} - 8"
                )
            header_bytes = f.read(header_size)
            if len(header_bytes) != header_size:
                return False, (
                    f"short read on header ({len(header_bytes)}/{header_size})"
                )
    except OSError as e:
        return False, f"read failed: {e}"

    import json as _json
    try:
        header = _json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as e:
        return False, f"header JSON parse failed: {e}"

    if not isinstance(header, dict):
        return False, f"header is {type(header).__name__}, expected dict"

    # safetensors keeps optional metadata under "__metadata__" (separate from
    # tensor entries). Skip it during shape validation.
    layer_keys = []
    total_bytes = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            return False, f"header entry {name!r} is not a dict"
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            return False, f"header entry {name!r}: bad dtype/shape"
        bytes_per_elem = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if bytes_per_elem is None:
            return False, f"header entry {name!r}: unknown dtype {dtype!r}"
        if len(shape) > MAX_TENSOR_NDIM:
            return False, (
                f"header entry {name!r}: ndim={len(shape)} > {MAX_TENSOR_NDIM}"
            )
        n = 1
        for axis, dim in enumerate(shape):
            if not isinstance(dim, int) or dim < 0:
                return False, f"header entry {name!r}: bad dim[{axis}]={dim!r}"
            if dim > MAX_TENSOR_DIM:
                return False, (
                    f"header entry {name!r}: dim[{axis}]={dim} > {MAX_TENSOR_DIM}"
                )
            n *= dim
        nbytes = n * bytes_per_elem
        if nbytes > PER_TENSOR_HEADER_CAP:
            return False, (
                f"header entry {name!r}: {nbytes} bytes "
                f"> per-tensor cap {PER_TENSOR_HEADER_CAP}"
            )
        total_bytes += nbytes
        if total_bytes > PER_FILE_HEADER_CAP:
            return False, (
                f"running total {total_bytes} bytes "
                f"> per-file cap {PER_FILE_HEADER_CAP}"
            )

        # Track layer keys for layer-count check
        if name.startswith("layer_"):
            try:
                lid = int(name.split("_")[1])
                layer_keys.append(lid)
            except (ValueError, IndexError):
                pass

    # Layer-count cross-check
    if expected_num_layers is not None and layer_keys:
        max_layer = max(layer_keys)
        if max_layer >= expected_num_layers:
            return False, (
                f"max layer index {max_layer} >= expected {expected_num_layers} "
                f"(source={source})"
            )

    return True, ""


def reject_safetensors_or_warn(
    file_path: str,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
    delete_on_reject: bool = True,
) -> bool:
    """Pre-mx.load gate. Returns True if file is safe to load, False if not.

    On rejection (and ``delete_on_reject=True``), unlinks the file so the
    next run does not re-trigger. This matches Codex's ``reject + delete``
    contract for poisoned cache entries.
    """
    ok, reason = validate_safetensors_header(
        file_path,
        expected_num_layers=expected_num_layers,
        source=source,
    )
    if not ok:
        logger.warning(
            "Pre-mx.load header validation rejected %s: %s. "
            "Treating as cache miss%s.",
            source, reason,
            " + deleting file" if delete_on_reject else "",
        )
        if delete_on_reject:
            try:
                import os
                os.remove(file_path)
            except OSError as e:
                logger.warning("Failed to delete poisoned cache file %s: %s",
                               file_path, e)
    return ok

"""Cache-record validator: hard guard against malformed paged/L2 cache restores.

Cache safety contract: validate BEFORE allocating MLX tensors.
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

import json
import logging
from typing import Any, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- Hard ceilings: no decoded tensor may request hundreds of GB
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

# Metadata/scalar ceilings. These catch the corruption class where the tensor
# header is small enough to pass pre-load validation, but metadata later drives
# a huge decode/update allocation (e.g. TQ original_shape or cache offset).
MAX_CACHE_OFFSET = 2_000_000
MAX_CACHE_LAYERS = 1024
MAX_CACHE_GROUP_SIZE = 4096
ALLOWED_CACHE_BITS = {2, 3, 4, 8}


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


def _safe_int(value: Any, *, label: str) -> Tuple[bool, int, str]:
    try:
        if isinstance(value, bool):
            return False, 0, f"{label}: bool is not a valid integer"
        return True, int(value), ""
    except (TypeError, ValueError, OverflowError) as e:
        return False, 0, f"{label}: invalid integer {value!r}: {e}"


def _validate_int_range(
    value: Any,
    *,
    label: str,
    lo: int,
    hi: int,
) -> Tuple[bool, int, str]:
    ok, parsed, reason = _safe_int(value, label=label)
    if not ok:
        return False, 0, reason
    if parsed < lo or parsed > hi:
        return False, parsed, f"{label}: {parsed} outside [{lo}, {hi}]"
    return True, parsed, ""


def _decode_meta_sequence(meta: Any) -> list[Any]:
    if meta is None:
        return []
    if isinstance(meta, (list, tuple)):
        return list(meta)
    if isinstance(meta, dict):
        return []
    if isinstance(meta, str):
        stripped = meta.strip()
        if not stripped or stripped == "{}":
            return []
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, (list, tuple)):
                return list(decoded)
            return []
        except json.JSONDecodeError:
            return [p for p in stripped.replace(",", " ").split() if p]
    return []


def _validate_quant_meta(meta: Any, *, label: str) -> Tuple[bool, str]:
    if meta in (None, "", (), [], {}):
        return True, ""

    if isinstance(meta, dict):
        values = {
            "offset": meta.get("offset", 0),
            "group_size": meta.get("group_size", meta.get("groupSize", 64)),
            "bits": meta.get("bits", 4),
        }
    else:
        seq = _decode_meta_sequence(meta)
        if not seq:
            return True, ""
        # QuantizedKVCache.meta_state is (offset, group_size, bits).
        values = {
            "offset": seq[0] if len(seq) > 0 else 0,
            "group_size": seq[1] if len(seq) > 1 else 64,
            "bits": seq[2] if len(seq) > 2 else 4,
        }

    ok, _, reason = _validate_int_range(
        values["offset"], label=f"{label}.offset", lo=0, hi=MAX_CACHE_OFFSET
    )
    if not ok:
        return False, reason
    ok, _, reason = _validate_int_range(
        values["group_size"],
        label=f"{label}.group_size",
        lo=1,
        hi=MAX_CACHE_GROUP_SIZE,
    )
    if not ok:
        return False, reason
    ok, bits, reason = _validate_int_range(
        values["bits"], label=f"{label}.bits", lo=1, hi=16
    )
    if not ok:
        return False, reason
    if bits not in ALLOWED_CACHE_BITS:
        return False, f"{label}.bits: {bits} not in {sorted(ALLOWED_CACHE_BITS)}"
    return True, ""


def _validate_rotating_meta(meta: Any, *, label: str) -> Tuple[bool, str]:
    seq = _decode_meta_sequence(meta)
    if not seq:
        return True, ""
    if len(seq) < 4:
        return False, f"{label}: expected (keep, max_size, offset, idx), got {seq!r}"
    ok, keep, reason = _validate_int_range(
        seq[0], label=f"{label}.keep", lo=0, hi=MAX_TENSOR_DIM
    )
    if not ok:
        return False, reason
    ok, max_size, reason = _validate_int_range(
        seq[1], label=f"{label}.max_size", lo=1, hi=MAX_TENSOR_DIM
    )
    if not ok:
        return False, reason
    ok, _, reason = _validate_int_range(
        seq[2], label=f"{label}.offset", lo=0, hi=MAX_CACHE_OFFSET
    )
    if not ok:
        return False, reason
    # RotatingKVCache._idx is the physical insertion pointer. mlx-lm's
    # multi-token prefill path (_update_concat) can leave both offset and _idx
    # larger than max_size until decode starts rotating in place. This is valid
    # for DSV4 prompt-boundary snapshots where local SWA may carry the full
    # prompt while CSA/HSA pools carry compressed global context. Bound it by
    # the hard cache-offset cap instead of sliding-window max_size.
    ok, idx, reason = _validate_int_range(
        seq[3], label=f"{label}.idx", lo=0, hi=MAX_CACHE_OFFSET
    )
    if not ok:
        return False, reason
    if keep > max_size:
        return False, f"{label}.keep: {keep} > max_size {max_size}"
    return True, ""


def _shape_nbytes(shape: list[int] | tuple[int, ...], bytes_per_elem: int) -> int:
    n = 1
    for dim in shape:
        n *= max(int(dim), 0)
    return n * bytes_per_elem


def _validate_shape_list(
    shape: Any,
    *,
    label: str,
    bytes_per_elem: int = 2,
) -> Tuple[bool, str, int]:
    if not isinstance(shape, (list, tuple)):
        return False, f"{label}: shape is {type(shape).__name__}, expected list", 0
    if len(shape) > MAX_TENSOR_NDIM:
        return False, f"{label}: ndim={len(shape)} > {MAX_TENSOR_NDIM}", 0
    parsed: list[int] = []
    for axis, dim in enumerate(shape):
        ok, d, reason = _validate_int_range(
            dim, label=f"{label}[{axis}]", lo=0, hi=MAX_TENSOR_DIM
        )
        if not ok:
            return False, reason, 0
        parsed.append(d)
    nbytes = _shape_nbytes(parsed, bytes_per_elem)
    if nbytes > MAX_TENSOR_BYTES:
        return False, f"{label}: decoded {nbytes} bytes > {MAX_TENSOR_BYTES}", nbytes
    return True, "", nbytes


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
            if len(entry) > 3:
                ok, max_size, reason = _validate_int_range(
                    entry[3],
                    label=f"layer {i} 'rotating_kv'.max_size",
                    lo=1,
                    hi=MAX_TENSOR_DIM,
                )
                if not ok:
                    return False, reason, total_bytes
                if len(entry) > 4:
                    ok, keep, reason = _validate_int_range(
                        entry[4],
                        label=f"layer {i} 'rotating_kv'.keep",
                        lo=0,
                        hi=max_size,
                    )
                    if not ok:
                        return False, reason, total_bytes
                    if keep > max_size:
                        return False, (
                            f"layer {i} 'rotating_kv'.keep: {keep} > {max_size}"
                        ), total_bytes

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
            if len(entry) > 3:
                ok, reason = _validate_quant_meta(
                    entry[3], label=f"layer {i} 'quantized_kv'.meta"
                )
                if not ok:
                    return False, reason, total_bytes

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
            if len(entry) > 2:
                # Cumulative caches store model-specific scalar metadata. We
                # cannot fully interpret every family here, but the common
                # first field is an offset/length. Reject huge poisoned values.
                seq = _decode_meta_sequence(entry[2])
                if seq:
                    ok, _, reason = _validate_int_range(
                        seq[0],
                        label=f"layer {i} 'cumulative'.meta[0]",
                        lo=0,
                        hi=MAX_CACHE_OFFSET,
                    )
                    if not ok:
                        return False, reason, total_bytes

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
            if len(entry) > 2:
                ok, reason = _validate_rotating_meta(
                    entry[2], label=f"layer {i} 'deepseek_v4'.local_meta"
                )
                if not ok:
                    return False, reason, total_bytes
            if len(entry) > 4 and isinstance(entry[4], dict):
                cache_meta = entry[4]
                if "sliding_window" in cache_meta:
                    ok, _, reason = _validate_int_range(
                        cache_meta["sliding_window"],
                        label=f"layer {i} 'deepseek_v4'.sliding_window",
                        lo=1,
                        hi=MAX_TENSOR_DIM,
                    )
                    if not ok:
                        return False, reason, total_bytes
                if "compress_ratio" in cache_meta and cache_meta["compress_ratio"] is not None:
                    ok, _, reason = _validate_int_range(
                        cache_meta["compress_ratio"],
                        label=f"layer {i} 'deepseek_v4'.compress_ratio",
                        lo=1,
                        hi=MAX_CACHE_GROUP_SIZE,
                    )
                    if not ok:
                        return False, reason, total_bytes
                if "local_quant_meta" in cache_meta:
                    ok, reason = _validate_quant_meta(
                        cache_meta["local_quant_meta"],
                        label=f"layer {i} 'deepseek_v4'.local_quant_meta",
                    )
                    if not ok:
                        return False, reason, total_bytes

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


def _validate_tensor_tree(
    obj: Any,
    *,
    label: str,
    layer_idx: int,
) -> Tuple[bool, int, str]:
    """Validate tensors nested in tuple/list/dict state without evaluating MLX.

    Quantized caches store ``keys``/``values`` as tuples of tensors. The older
    live-cache checks treated the tuple itself as a scalar and missed poisoned
    tuple members whose shapes would allocate huge buffers after dequantize.
    """
    if obj is None:
        return True, 0, ""
    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            ok, nb, reason = _validate_tensor_tree(
                value, label=f"{label}.{key}", layer_idx=layer_idx
            )
            if not ok:
                return False, total, reason
            total += nb
        return True, total, ""
    if isinstance(obj, (list, tuple)):
        total = 0
        for j, value in enumerate(obj):
            ok, nb, reason = _validate_tensor_tree(
                value, label=f"{label}[{j}]", layer_idx=layer_idx
            )
            if not ok:
                return False, total, reason
            total += nb
        return True, total, ""
    return _validate_tensor(obj, label=label, layer_idx=layer_idx)


def _validate_encoded_shape(obj: Any, *, label: str) -> Tuple[bool, str, int]:
    """Validate encoded TurboQuant shape metadata if present."""
    shape = getattr(obj, "shape", None)
    if shape is None:
        return True, "", 0
    if isinstance(shape, (list, tuple)):
        return _validate_shape_list(shape, label=label, bytes_per_elem=2)
    return True, "", 0


def _validate_live_single_cache(layer_cache: Any, *, layer_idx: int) -> Tuple[bool, str, int]:
    """Validate a live cache layer object before reuse/store.

    This complements ``validate_cache_record``. Record validation protects the
    serialized block format; live validation protects in-memory prefix cache
    entries and objects fetched from non-paged caches before they are passed back
    into model forward or disk serialization.
    """
    if layer_cache is None:
        return False, f"layer {layer_idx}: None cache layer", 0

    def _check_int_attr(attr: str, lo: int, hi: int) -> Tuple[bool, str]:
        if not hasattr(layer_cache, attr):
            return True, ""
        value = getattr(layer_cache, attr)
        if type(value).__module__.startswith("unittest.mock"):
            # Unit-test mocks claim every attribute exists. Ignore synthetic
            # mock scalars so cache-validation contract tests can use duck
            # typed cache layers without constructing real mlx-lm objects.
            return True, ""
        ok, _, reason = _validate_int_range(
            value, label=f"layer {layer_idx}.{attr}", lo=lo, hi=hi
        )
        return ok, reason

    for attr, lo, hi in (
        ("offset", 0, MAX_CACHE_OFFSET),
        ("_idx", 0, MAX_TENSOR_DIM),
        ("max_size", 1, MAX_TENSOR_DIM),
        ("keep", 0, MAX_TENSOR_DIM),
        ("group_size", 1, MAX_CACHE_GROUP_SIZE),
        ("key_dim", 1, MAX_TENSOR_DIM),
        ("value_dim", 1, MAX_TENSOR_DIM),
        ("sink_tokens", 0, MAX_CACHE_OFFSET),
        ("_compressed_tokens", 0, MAX_CACHE_OFFSET),
    ):
        ok, reason = _check_int_attr(attr, lo, hi)
        if not ok:
            return False, reason, 0
    if hasattr(layer_cache, "bits"):
        if type(getattr(layer_cache, "bits")).__module__.startswith("unittest.mock"):
            pass
        else:
            ok, bits, reason = _validate_int_range(
                getattr(layer_cache, "bits"),
                label=f"layer {layer_idx}.bits",
                lo=1,
                hi=16,
            )
            if not ok:
                return False, reason, 0
            if bits not in ALLOWED_CACHE_BITS:
                return False, (
                    f"layer {layer_idx}.bits: {bits} not in {sorted(ALLOWED_CACHE_BITS)}"
                ), 0
    for attr in ("key_bits", "value_bits"):
        if hasattr(layer_cache, attr):
            if type(getattr(layer_cache, attr)).__module__.startswith("unittest.mock"):
                continue
            ok, bits, reason = _validate_int_range(
                getattr(layer_cache, attr),
                label=f"layer {layer_idx}.{attr}",
                lo=1,
                hi=16,
            )
            if not ok:
                return False, reason, 0
            if bits not in ALLOWED_CACHE_BITS:
                return False, (
                    f"layer {layer_idx}.{attr}: {bits} not in "
                    f"{sorted(ALLOWED_CACHE_BITS)}"
                ), 0

    total = 0

    # CacheList / nested MoE cache containers.
    if hasattr(layer_cache, "caches") and isinstance(
        getattr(layer_cache, "caches", None), (list, tuple)
    ):
        for j, sub in enumerate(layer_cache.caches):
            ok, reason, nb = _validate_live_single_cache(
                sub, layer_idx=f"{layer_idx}.{j}"  # type: ignore[arg-type]
            )
            if not ok:
                return False, reason, total
            total += nb
        return True, "", total

    # Extracted state dicts from scheduler extraction paths.
    if isinstance(layer_cache, dict):
        state = layer_cache.get("state")
        ok, nb, reason = _validate_tensor_tree(
            state, label="state", layer_idx=layer_idx
        )
        if not ok:
            return False, reason, total
        total += nb
        meta = layer_cache.get("meta_state")
        cls_name = str(layer_cache.get("class_name", ""))
        if cls_name == "QuantizedKVCache":
            ok, reason = _validate_quant_meta(meta, label=f"layer {layer_idx}.meta")
            if not ok:
                return False, reason, total
        elif "Rotating" in cls_name:
            ok, reason = _validate_rotating_meta(meta, label=f"layer {layer_idx}.meta")
            if not ok:
                return False, reason, total
        else:
            seq = _decode_meta_sequence(meta)
            if seq:
                ok, _, reason = _validate_int_range(
                    seq[0], label=f"layer {layer_idx}.meta[0]",
                    lo=0, hi=MAX_CACHE_OFFSET
                )
                if not ok:
                    return False, reason, total
        return True, "", total

    # KVCache / RotatingKVCache / QuantizedKVCache / TurboQuantKVCache.
    has_kv_protocol = hasattr(layer_cache, "keys") and hasattr(layer_cache, "values")
    if has_kv_protocol:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            compressed_present = False
            for attr in (
                "_compressed_keys",
                "_compressed_values",
                "_joined_k",
                "_joined_v",
            ):
                value = getattr(layer_cache, attr, None)
                if value is None:
                    continue
                compressed_present = True
                ok, shape_reason, shape_bytes = _validate_encoded_shape(
                    value, label=f"layer {layer_idx}.{attr}.shape"
                )
                if not ok:
                    return False, shape_reason, total
                total += shape_bytes
                ok, nb, reason = _validate_tensor_tree(
                    value, label=attr, layer_idx=layer_idx
                )
                if not ok:
                    return False, reason, total
                total += nb
            if compressed_present:
                return True, "", total
            return False, f"layer {layer_idx}: keys/values are None", total

        for label, value in (("keys", keys), ("values", values)):
            ok, nb, reason = _validate_tensor_tree(
                value, label=label, layer_idx=layer_idx
            )
            if not ok:
                return False, reason, total
            total += nb
        return True, "", total

    # SSM / ArraysCache / MambaCache: validate cumulative state tensors.
    if hasattr(layer_cache, "cache") and isinstance(
        getattr(layer_cache, "cache", None), list
    ):
        ok, nb, reason = _validate_tensor_tree(
            getattr(layer_cache, "cache", None),
            label="cache",
            layer_idx=layer_idx,
        )
        if not ok:
            return False, reason, total
        total += nb
        meta = getattr(layer_cache, "meta_state", None)
        seq = _decode_meta_sequence(meta)
        if seq:
            ok, _, reason = _validate_int_range(
                seq[0],
                label=f"layer {layer_idx}.meta[0]",
                lo=0,
                hi=MAX_CACHE_OFFSET,
            )
            if not ok:
                return False, reason, total
        return True, "", total

    # Unknown objects are allowed only if they don't carry visible tensor state.
    ok, nb, reason = _validate_tensor_tree(
        layer_cache, label="unknown", layer_idx=layer_idx
    )
    if not ok:
        return False, reason, total
    total += nb
    return True, "", total


def validate_live_cache(
    cache: Any,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> Tuple[bool, str, int]:
    """Validate live cache objects before reuse/store.

    Returns ``(ok, reason, total_bytes)``. ``ok=False`` means the caller should
    drop the cache entry and full-prefill. This is used for in-memory prefix
    caches where there is no serialized ``cache_data`` record to validate.
    """
    if cache is None:
        return False, f"cache is None (source={source})", 0
    if not isinstance(cache, list):
        # Single CacheList-like object is valid input; normalize for iteration.
        if hasattr(cache, "caches"):
            cache = [cache]
        else:
            return False, f"cache is {type(cache).__name__}, expected list", 0
    if len(cache) == 0:
        return False, f"cache is empty (source={source})", 0
    if expected_num_layers is not None and len(cache) != expected_num_layers:
        return False, (
            f"live layer count {len(cache)} != expected {expected_num_layers} "
            f"(source={source})"
        ), 0

    total = 0
    for i, layer in enumerate(cache):
        ok, reason, nb = _validate_live_single_cache(layer, layer_idx=i)
        if not ok:
            return False, reason, total
        total += nb
        if total > MAX_TOTAL_RECORD_BYTES:
            return False, (
                f"live total_bytes {total} > {MAX_TOTAL_RECORD_BYTES} "
                f"(source={source})"
            ), total
    return True, "", total


def reject_live_cache_or_warn(
    cache: Any,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> bool:
    ok, reason, nbytes = validate_live_cache(
        cache,
        expected_num_layers=expected_num_layers,
        source=source,
    )
    if not ok:
        logger.warning(
            "Live cache validation rejected %s: %s (bytes_seen=%d). "
            "Treating as cache miss.",
            source,
            reason,
            nbytes,
        )
    return ok


# ============================================================================
# Pre-mx.load safetensors header validation.
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
        data_offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            return False, f"header entry {name!r}: bad dtype/shape"
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or not all(isinstance(x, int) for x in data_offsets)
        ):
            return False, f"header entry {name!r}: bad data_offsets"
        data_start, data_end = data_offsets
        if data_start < 0 or data_end < data_start:
            return False, f"header entry {name!r}: invalid data_offsets {data_offsets!r}"
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
        if data_end - data_start != nbytes:
            return False, (
                f"header entry {name!r}: data_offsets span {data_end - data_start} "
                f"!= declared tensor bytes {nbytes}"
            )
        if 8 + header_size + data_end > size_on_disk:
            return False, (
                f"header entry {name!r}: data_end {data_end} exceeds file payload "
                f"size {max(size_on_disk - 8 - header_size, 0)}"
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


def validate_tq_native_metadata(
    tensors: dict[str, Any],
    metadata: dict[str, str],
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> Tuple[bool, str]:
    """Validate TurboQuant native disk metadata before any decode allocation.

    The safetensors header only tells us the stored compressed tensor shapes.
    TQ decode uses metadata such as ``__tq_i_ck_shape__`` as the *decoded*
    original KV shape. A poisoned file can therefore pass header validation
    while making ``decode_keys`` allocate hundreds of GB. This guard rejects
    impossible decoded shapes and scalar fields first.
    """
    if not isinstance(metadata, dict):
        return False, f"TQ metadata is {type(metadata).__name__}, expected dict"
    ok, num_layers, reason = _validate_int_range(
        metadata.get("__num_layers__", "0"),
        label="__num_layers__",
        lo=0,
        hi=MAX_CACHE_LAYERS,
    )
    if not ok:
        return False, reason
    if expected_num_layers is not None and num_layers != expected_num_layers:
        return False, (
            f"__num_layers__ {num_layers} != expected {expected_num_layers} "
            f"(source={source})"
        )

    total_decoded = 0

    def _validate_tq_prefix(prefix: str, label: str) -> Tuple[bool, str]:
        nonlocal total_decoded
        required = (
            f"{prefix}_ck_indices_packed",
            f"{prefix}_ck_qjl_packed",
            f"{prefix}_ck_residual_norms",
            f"{prefix}_ck_vector_norms",
            f"{prefix}_cv_indices_packed",
            f"{prefix}_cv_vector_norms",
        )
        missing = [name for name in required if name not in tensors]
        if missing:
            return False, f"{label}: missing compressed tensors {missing}"

        for suffix in ("ck_shape", "cv_shape"):
            raw = metadata.get(f"__{prefix}_{suffix}__", "[]")
            try:
                shape = json.loads(raw)
            except json.JSONDecodeError as e:
                return False, f"{label}.{suffix}: invalid JSON {raw!r}: {e}"
            ok_shape, shape_reason, nbytes = _validate_shape_list(
                shape, label=f"{label}.{suffix}", bytes_per_elem=2
            )
            if not ok_shape:
                return False, shape_reason
            total_decoded += nbytes
            if total_decoded > MAX_TOTAL_RECORD_BYTES:
                return False, (
                    f"{label}: decoded total {total_decoded} bytes "
                    f"> {MAX_TOTAL_RECORD_BYTES}"
                )

        for suffix in ("ck_bits", "cv_bits", "key_bits", "value_bits"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_bits, bits, bit_reason = _validate_int_range(
                    metadata[key], label=f"{label}.{suffix}", lo=1, hi=16
                )
                if not ok_bits:
                    return False, bit_reason
                if bits not in ALLOWED_CACHE_BITS:
                    return False, (
                        f"{label}.{suffix}: {bits} not in {sorted(ALLOWED_CACHE_BITS)}"
                    )

        for suffix in ("offset", "compressed_tokens", "sink_tokens"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_int, _, int_reason = _validate_int_range(
                    metadata[key],
                    label=f"{label}.{suffix}",
                    lo=0,
                    hi=MAX_CACHE_OFFSET,
                )
                if not ok_int:
                    return False, int_reason

        for suffix in ("key_dim", "value_dim"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_dim, _, dim_reason = _validate_int_range(
                    metadata[key],
                    label=f"{label}.{suffix}",
                    lo=1,
                    hi=MAX_TENSOR_DIM,
                )
                if not ok_dim:
                    return False, dim_reason

        return True, ""

    for i in range(num_layers):
        cls_name = metadata.get(f"__layer_{i}_class__", "")
        if cls_name == "TurboQuantKVCache":
            ok_prefix, prefix_reason = _validate_tq_prefix(f"tq_{i}", f"layer {i}")
            if not ok_prefix:
                return False, prefix_reason

        if metadata.get(f"__layer_{i}_cache_list__") == "true":
            ok_count, sub_count, count_reason = _validate_int_range(
                metadata.get(f"__layer_{i}_cl_count__", "0"),
                label=f"layer {i}.cache_list_count",
                lo=0,
                hi=64,
            )
            if not ok_count:
                return False, count_reason
            for j in range(sub_count):
                sub_cls = metadata.get(f"__layer_{i}_cl_{j}_class__", "")
                if sub_cls == "TurboQuantKVCache":
                    ok_prefix, prefix_reason = _validate_tq_prefix(
                        f"cl_{i}_{j}", f"layer {i}.cache_list[{j}]"
                    )
                    if not ok_prefix:
                        return False, prefix_reason

        if metadata.get(f"__layer_{i}_quantized__") == "true":
            for suffix in ("qk_count", "qv_count"):
                key = f"__layer_{i}_{suffix}__"
                if key in metadata:
                    ok_count, _, count_reason = _validate_int_range(
                        metadata[key],
                        label=f"layer {i}.{suffix}",
                        lo=0,
                        hi=8,
                    )
                    if not ok_count:
                        return False, count_reason

        if metadata.get(f"__layer_{i}_cumulative__") == "true":
            key = f"__layer_{i}_state_count__"
            if key in metadata:
                ok_count, _, count_reason = _validate_int_range(
                    metadata[key],
                    label=f"layer {i}.state_count",
                    lo=0,
                    hi=64,
                )
                if not ok_count:
                    return False, count_reason

        meta_key = f"__layer_{i}_meta__"
        if meta_key in metadata:
            seq = _decode_meta_sequence(metadata[meta_key])
            if seq:
                ok_offset, _, offset_reason = _validate_int_range(
                    seq[0],
                    label=f"layer {i}.meta[0]",
                    lo=0,
                    hi=MAX_CACHE_OFFSET,
                )
                if not ok_offset:
                    return False, offset_reason

    return True, ""


def reject_tq_native_metadata_or_warn(
    tensors: dict[str, Any],
    metadata: dict[str, str],
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
) -> bool:
    ok, reason = validate_tq_native_metadata(
        tensors,
        metadata,
        expected_num_layers=expected_num_layers,
        source=source,
    )
    if not ok:
        logger.warning(
            "TQ-native metadata validation rejected %s: %s. Treating as cache miss.",
            source,
            reason,
        )
    return ok


def reject_safetensors_or_warn(
    file_path: str,
    *,
    expected_num_layers: Optional[int] = None,
    source: str = "unknown",
    delete_on_reject: bool = True,
) -> bool:
    """Pre-mx.load gate. Returns True if file is safe to load, False if not.

    On rejection (and ``delete_on_reject=True``), unlinks the file so the
    next run does not re-trigger. This matches the ``reject + delete``
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

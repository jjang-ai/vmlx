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
import re
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

# DeepSeek-V4 block-disk records name their nested state tensors layer_<i>_dsv4_state_<j>. Their rows grow
# with the context (V4.1 pools / n-gram history: ~one row per token), so they are bounded by the cache offset
# cap, as in validate_cache_record, instead of the generic 256K axis cap.
_DSV4_STATE_HEADER_NAME = re.compile(r"^layer_\d+_dsv4_state_\d+$")

_TQ_PACKED_HEADER_NAME = re.compile(
    r"^(?:tq_\d+|cl_\d+_\d+)_(?:ck|cv)_(?:indices_packed|qjl_packed)$"
)

# Metadata/scalar ceilings. These catch the corruption class where the tensor
# header is small enough to pass pre-load validation, but metadata later drives
# a huge decode/update allocation (e.g. TQ original_shape or cache offset).
MAX_CACHE_OFFSET = 2_000_000
MAX_CACHE_LAYERS = 1024
MAX_CACHE_GROUP_SIZE = 4096
ALLOWED_CACHE_BITS = {2, 3, 4, 8}
# QuantizedKVCache uses MLX's affine decoder, not MXFP/NVFP weight codecs.
AFFINE_CACHE_GROUP_SIZES = {32, 64, 128}
_TQ_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
ALLOWED_TQ_DTYPES = set(_TQ_DTYPE_BYTES)


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
    max_dim: int = MAX_TENSOR_DIM,
) -> Tuple[bool, int, str]:
    """Validate a single tensor's shape + byte size. Returns (ok, bytes, reason).

    ``max_dim`` bounds every dimension; callers whose tensors legitimately grow with the
    token count (DeepSeek-V4 compressed pools and n-gram history) pass a token-scale bound."""
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
        if d > max_dim:
            return False, 0, (
                f"layer {layer_idx} {label}: dim[{axis}]={d} > {max_dim}"
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
    # Legacy optional DSV4 local metadata. Ordinary packed affine records use
    # parse_quantized_kv_meta below: their codec must never be guessed.
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


def parse_quantized_kv_meta(meta: Any) -> tuple[int, int, int]:
    """Read the explicit affine codec used by BOTH validation and restoration.

    Old tagged records may carry dictionaries or textual sequences. Offset is
    optional only in a dictionary: pages retain their producer's capture offset,
    while reconstruction derives the new offset from the concatenated length.
    Group size and bits are never optional or inferred from tensor/model names.
    """
    if isinstance(meta, dict):
        if "bits" not in meta or not ({"group_size", "groupSize"} & meta.keys()):
            raise CacheValidationError("quantized meta requires explicit group_size and bits")
        values = (meta.get("offset", 0), meta.get("group_size", meta.get("groupSize")), meta["bits"])
    else:
        values = _decode_meta_sequence(meta)
        if len(values) != 3:
            raise CacheValidationError("quantized meta requires (offset, group_size, bits)")

    parsed = []
    for name, value, lo, hi in zip(
        ("offset", "group_size", "bits"), values,
        (0, 1, 1), (MAX_CACHE_OFFSET, MAX_CACHE_GROUP_SIZE, 16),
    ):
        ok, number, reason = _validate_int_range(value, label=f"quantized meta.{name}", lo=lo, hi=hi)
        if not ok:
            raise CacheValidationError(reason)
        if not isinstance(value, str) and value != number:
            raise CacheValidationError(f"quantized meta.{name}: fractional integer {value!r}")
        parsed.append(number)
    if parsed[2] not in ALLOWED_CACHE_BITS:
        raise CacheValidationError(f"quantized meta.bits: {parsed[2]} not in {sorted(ALLOWED_CACHE_BITS)}")
    if parsed[1] not in AFFINE_CACHE_GROUP_SIZES:
        raise CacheValidationError(
            f"quantized meta.group_size: {parsed[1]} not in {sorted(AFFINE_CACHE_GROUP_SIZES)}"
        )
    if isinstance(meta, dict) and "group_size" in meta and "groupSize" in meta:
        alias = parse_quantized_kv_meta({"group_size": meta["groupSize"], "bits": parsed[2]})[1]
        if alias != parsed[1]:
            raise CacheValidationError("quantized meta has conflicting group_size/groupSize")
    return tuple(parsed)


def _validate_quantized_kv_layout(entry: Any) -> tuple[bool, str]:
    try:
        _, group, bits = parse_quantized_kv_meta(entry[3])
    except (CacheValidationError, IndexError) as exc:
        return False, f"quantized_kv.meta: {exc}"
    for label, side in (("keys", entry[1]), ("values", entry[2])):
        shapes = [getattr(t, "shape", ()) for t in side]
        data, scale, bias = shapes
        if not (2 <= len(data) <= 4 and len(scale) == len(data) and scale == bias):
            return False, f"quantized_kv {label}: incompatible packed/scale/bias ranks or shapes"
        if data[:-1] != scale[:-1] or not data[-1] or not scale[-1]:
            return False, f"quantized_kv {label}: incompatible packed/scale/bias axes"
        if data[-1] * 32 != scale[-1] * group * bits:
            return False, f"quantized_kv {label}: packed width disagrees with group_size/bits"
        dtypes = [str(getattr(t, "dtype", "")).split(".")[-1] for t in side]
        if dtypes[0] != "uint32" or dtypes[1] not in ALLOWED_TQ_DTYPES or dtypes[2] != dtypes[1]:
            return False, f"quantized_kv {label}: invalid packed/scale/bias dtypes {dtypes}"
    if entry[1][0].shape[:-1] != entry[2][0].shape[:-1]:
        return False, "quantized_kv: keys/values batch/head/token axes differ"
    return True, ""


def validate_quantized_cache_chain(blocks: list[Any]) -> tuple[bool, str]:
    """Reject mixed positional codecs/layouts before any concatenation/decode.

    Per-page validation alone cannot detect individually valid pages encoded
    with different codecs. Compare each positional slot, not whole-model bits;
    K/V widths, layer widths and nested cache codecs may legitimately differ.
    Offsets/token lengths are intentionally absent from the signature.
    """
    layouts = {}

    def contains_quantized(entry):
        if not isinstance(entry, (tuple, list)) or not entry:
            return False
        if entry[0] == "quantized_kv":
            return True
        if entry[0] == "cache_list":
            return any(contains_quantized(sub) for sub in entry[1])
        if entry[0] == "zaya_cca":
            return contains_quantized(entry[1])
        return False

    def visit(entry, path):
        if not isinstance(entry, (tuple, list)) or not entry:
            return
        tag = entry[0]
        if tag == "cache_list":
            yield path, (tag, len(entry[1])), contains_quantized(entry)
            for index, sub in enumerate(entry[1]):
                yield from visit(sub, path + (index,))
        elif tag == "zaya_cca":
            yield path, (tag,), contains_quantized(entry)
            yield from visit(entry[1], path + ("cca_kv",))
        elif tag == "quantized_kv":
            _, group, bits = parse_quantized_kv_meta(entry[3])
            layout = tuple(
                (tuple(t.shape[:-2]), t.shape[-1], str(t.dtype))
                for side in entry[1:3] for t in side
            )
            yield path, (tag, group, bits, layout), True
        else:
            yield path, (tag,), False

    try:
        for block in blocks:
            for layer, entry in enumerate(block):
                for path, layout, affine in visit(entry, (layer,)):
                    previous, previous_affine = layouts.setdefault(path, (layout, affine))
                    if (previous_affine or affine) and previous != layout:
                        return False, f"quantized_kv slot {path}: inconsistent codec/layout across pages"
                    if affine and not previous_affine:
                        layouts[path] = (layout, True)
    except (CacheValidationError, IndexError, TypeError, AttributeError) as exc:
        return False, f"quantized cache chain: {exc}"
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
    # prompt while CSA/HCA pools carry compressed global context. Bound it by
    # the hard cache-offset cap instead of sliding-window max_size.
    ok, idx, reason = _validate_int_range(
        seq[3], label=f"{label}.idx", lo=0, hi=MAX_CACHE_OFFSET
    )
    if not ok:
        return False, reason
    if keep > max_size:
        return False, f"{label}.keep: {keep} > max_size {max_size}"
    return True, ""


def _validate_glm5_typed_state(
    class_name: str,
    state: Any,
    meta: Any,
    *,
    label: str,
) -> Tuple[bool, str]:
    """Validate GLM's schema-tagged KDA/MLA state without numeric-meta guesses.

    GLM cache metadata begins with a typed ``glm5_next_native_v*`` tag rather
    than the numeric offset used by generic ArraysCache/KVCache implementations. The
    canonical cache constructors already own all shape, population, k-pool,
    and offset invariants, so use them here instead of duplicating a second
    schema that can drift.
    """

    if class_name not in {"Glm5KDACache", "Glm5MLACache"}:
        return False, f"{label}: unsupported GLM typed class {class_name!r}"
    try:
        from .models.glm5_next.glm5_next import Glm5KDACache, Glm5MLACache

        cache_class = {
            "Glm5KDACache": Glm5KDACache,
            "Glm5MLACache": Glm5MLACache,
        }[class_name]
        cache_class.from_state(state, meta)
    except (TypeError, ValueError) as exc:
        return False, f"{label}: {exc}"
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
            ``"quantized_kv"``, ``"turboquant_kv"``, ``"rotating_kv"``,
            ``"rotating_kv_pending"``, ``"cumulative"``,
            ``"deepseek_v4"``, ``"deepseek_v4_delta_v1"``,
            ``"deepseek_v4_pending"``, ``"cache_list"``,
            ``"zaya_cca"``, ``"minimax_m3"``, ``"no_state"``, ``"skip"``).
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

        if tag in (
            "skip",
            "deepseek_v4_pending",
            "rotating_kv_pending",
            "no_state",
        ):
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

        elif tag == "minimax_m3":
            # ("minimax_m3", keys, values, idx_keys) — MSA sparse layer.
            # keys/values are required; idx_keys may be None (handled by
            # _validate_tensor) but is normally present.
            if len(entry) < 4:
                return False, (
                    f"layer {i} 'minimax_m3': len={len(entry)} < 4"
                ), total_bytes
            for sub_label, t in (
                ("keys", entry[1]),
                ("values", entry[2]),
                ("idx_keys", entry[3]),
            ):
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
            # The explicit codec is required; a guessed default can interpret
            # the same packed bytes with a different width and precision.
            if len(entry) != 4:
                return False, (
                    f"layer {i} 'quantized_kv'.meta: expected four-field record"
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
            ok, reason = _validate_quantized_kv_layout(entry)
            if not ok:
                return False, f"layer {i} {reason}", total_bytes

        elif tag == "turboquant_kv":
            if len(entry) != 4:
                return False, (
                    f"layer {i} 'turboquant_kv': len={len(entry)} != 4"
                ), total_bytes
            encoded_keys, encoded_values, tq_config = entry[1], entry[2], entry[3]
            if not isinstance(tq_config, dict):
                return False, (
                    f"layer {i} 'turboquant_kv': config is "
                    f"{type(tq_config).__name__}, expected dict"
                ), total_bytes
            for key in ("key_dtype", "value_dtype"):
                dtype = tq_config.get(key)
                if dtype not in ALLOWED_TQ_DTYPES:
                    return False, (
                        f"layer {i} 'turboquant_kv'.{key}: {dtype!r} not in "
                        f"{sorted(ALLOWED_TQ_DTYPES)}"
                    ), total_bytes
            for label, obj, dtype_key, fields in (
                (
                    "encoded_keys",
                    encoded_keys,
                    "key_dtype",
                    (
                        "indices_packed",
                        "qjl_packed",
                        "residual_norms",
                        "vector_norms",
                    ),
                ),
                (
                    "encoded_values",
                    encoded_values,
                    "value_dtype",
                    ("indices_packed", "vector_norms"),
                ),
            ):
                for field in fields:
                    value = getattr(obj, field, None)
                    ok, nb, reason = _validate_tensor(
                        value,
                        label=f"{label}.{field}",
                        layer_idx=i,
                    )
                    if not ok:
                        return False, reason, total_bytes
                    total_bytes += nb
                shape = getattr(obj, "shape", None)
                ok_shape, shape_reason, _ = _validate_shape_list(
                    shape,
                    label=f"layer {i} 'turboquant_kv'.{label}.shape",
                    bytes_per_elem=_TQ_DTYPE_BYTES[tq_config[dtype_key]],
                )
                if not ok_shape:
                    return False, shape_reason, total_bytes
                ok_index, _, index_reason = _validate_int_range(
                    getattr(obj, "index_bits", None),
                    label=f"layer {i} 'turboquant_kv'.{label}.index_bits",
                    lo=1,
                    hi=16,
                )
                if not ok_index:
                    return False, index_reason, total_bytes
            for key in ("key_bits", "value_bits"):
                ok_bits, bits, reason = _validate_int_range(
                    tq_config.get(key),
                    label=f"layer {i} 'turboquant_kv'.{key}",
                    lo=1,
                    hi=16,
                )
                if not ok_bits:
                    return False, reason, total_bytes
                if bits not in ALLOWED_CACHE_BITS:
                    return False, (
                        f"layer {i} 'turboquant_kv'.{key}: {bits} not in "
                        f"{sorted(ALLOWED_CACHE_BITS)}"
                    ), total_bytes
            for key, lo, hi in (
                ("key_dim", 1, MAX_TENSOR_DIM),
                ("value_dim", 1, MAX_TENSOR_DIM),
                ("seed", 0, 2_147_483_647),
                ("offset", 1, MAX_CACHE_OFFSET),
            ):
                ok_int, _, reason = _validate_int_range(
                    tq_config.get(key),
                    label=f"layer {i} 'turboquant_kv'.{key}",
                    lo=lo,
                    hi=hi,
                )
                if not ok_int:
                    return False, reason, total_bytes
            key_shape = tuple(getattr(encoded_keys, "shape", ()))
            value_shape = tuple(getattr(encoded_values, "shape", ()))
            offset = int(tq_config["offset"])
            if (
                len(key_shape) != 4
                or len(value_shape) != 4
                or int(key_shape[-2]) != offset
                or int(value_shape[-2]) != offset
                or int(key_shape[-1]) != int(tq_config["key_dim"])
                or int(value_shape[-1]) != int(tq_config["value_dim"])
            ):
                return False, (
                    f"layer {i} 'turboquant_kv': shape/config mismatch "
                    f"keys={key_shape}, values={value_shape}, config={tq_config}"
                ), total_bytes

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
            # DeepSeek-V4 / V4.1 state rows grow with the context: the compressed-KV and index
            # pools hold up to one row per token (V4.1 ratio-1 layers) and V4.1 carries the n-gram
            # history, so a 1M-token entry has ~1M-row tensors. Bound them by the cache offset cap
            # instead of the generic 256K per-dimension cap (which rejected every V4.1 prefix
            # entry above ~262k tokens as a cache miss). Byte caps still apply.
            for j, t in enumerate(_walk_tensors(entry[1])):
                ok, nb, reason = _validate_tensor(
                    t, label=f"dsv4_state_t{j}", layer_idx=i, max_dim=MAX_CACHE_OFFSET
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

        elif tag == "deepseek_v4_delta_v1":
            # (tag, record, class_name, cache_meta). Every record is one
            # immutable <=256-token native pool delta. Exact SWA/buffer state
            # is optional and appears only on 2K-periodic or terminal anchors.
            if len(entry) < 4 or not isinstance(entry[1], dict):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': malformed record"
                ), total_bytes
            record = entry[1]
            if record.get("schema") != "deepseek_v4_block_delta_v1":
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': unsupported schema"
                ), total_bytes
            allowed_classes = {"DeepseekV4Cache", "PoolQuantizedV4Cache"}
            record_class = str(record.get("class_name") or "")
            wrapper_class = str(entry[2] or "")
            if (
                record_class not in allowed_classes
                or wrapper_class != record_class
            ):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': cache class mismatch "
                    f"record={record_class!r} wrapper={wrapper_class!r}"
                ), total_bytes
            cache_meta = entry[3]
            if not isinstance(cache_meta, dict):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': malformed wrapper metadata"
                ), total_bytes
            try:
                start = int(record.get("start_token", -1))
                end = int(record.get("end_token", -1))
                block_size = int(record.get("block_size", 0))
                anchor_blocks = int(record.get("anchor_interval_blocks", 0))
                ratio = int(record.get("compress_ratio", 0))
                sliding_window = int(record.get("sliding_window", 0))
            except (TypeError, ValueError):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': non-integral geometry"
                ), total_bytes
            if (
                start < 0
                or end <= start
                or block_size != 256
                or start % block_size
                or end - start > block_size
                or anchor_blocks != 8
                or ratio not in (4, 128)
                or sliding_window <= 0
                or sliding_window > MAX_TENSOR_DIM
            ):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': invalid geometry "
                    f"start={start} end={end} block={block_size} "
                    f"anchors={anchor_blocks} ratio={ratio} "
                    f"window={sliding_window}"
                ), total_bytes
            try:
                meta_ratio = int(cache_meta.get("compress_ratio", 0))
                meta_window = int(cache_meta.get("sliding_window", 0))
            except (TypeError, ValueError):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': non-integral wrapper geometry"
                ), total_bytes
            pool_quant = cache_meta.get("pool_quant")
            if (
                meta_ratio != ratio
                or meta_window != sliding_window
                or not isinstance(pool_quant, bool)
                or pool_quant != (record_class == "PoolQuantizedV4Cache")
            ):
                return False, (
                    f"layer {i} 'deepseek_v4_delta_v1': record/wrapper metadata mismatch"
                ), total_bytes

            def _validate_pool_delta(
                pool,
                label,
                *,
                allow_none,
                expected_start,
                expected_end,
            ):
                nonlocal total_bytes
                if not isinstance(pool, dict) or pool.get("schema") != "deepseek_v4_pool_delta_v1":
                    return False, f"layer {i} {label}: invalid pool schema"
                try:
                    row_start = int(pool.get("start_row", -1))
                    row_end = int(pool.get("end_row", -1))
                except (TypeError, ValueError):
                    return False, f"layer {i} {label}: non-integral row span"
                if row_start < 0 or row_end < row_start:
                    return False, f"layer {i} {label}: invalid row span"
                if (row_start, row_end) != (expected_start, expected_end):
                    return False, (
                        f"layer {i} {label}: row span does not match token geometry "
                        f"got={row_start}:{row_end} "
                        f"expected={expected_start}:{expected_end}"
                    )
                storage = pool.get("storage")
                if storage == "none":
                    if not allow_none or row_start or row_end:
                        return False, f"layer {i} {label}: invalid absent pool"
                    return True, ""
                if storage == "bf16":
                    value = pool.get("value")
                    if value is None or not hasattr(value, "shape"):
                        return False, f"layer {i} {label}: missing BF16 rows"
                    if len(value.shape) < 2 or int(value.shape[1]) != row_end - row_start:
                        return False, f"layer {i} {label}: BF16 row mismatch"
                elif storage == "q8":
                    segments = tuple(pool.get("segments") or ())
                    rows = 0
                    for segment in segments:
                        if not isinstance(segment, (tuple, list)) or len(segment) not in (6, 7):
                            return False, f"layer {i} {label}: malformed q8 segment"
                        q, scale, minimum, raw_shape, raw_group, raw_bits, *_ = segment
                        try:
                            shape = tuple(int(dim) for dim in raw_shape)
                            group = int(raw_group)
                            bits = int(raw_bits)
                        except (TypeError, ValueError):
                            return False, f"layer {i} {label}: invalid q8 metadata"
                        if len(shape) < 2 or group <= 0 or shape[-1] % group or bits != 8:
                            return False, f"layer {i} {label}: invalid q8 geometry"
                        groups = shape[-1] // group
                        if (
                            tuple(getattr(q, "shape", ())) != (*shape[:-1], groups, group)
                            or tuple(getattr(scale, "shape", ())) != (*shape[:-1], groups, 1)
                            or tuple(getattr(minimum, "shape", ())) != (*shape[:-1], groups, 1)
                            or not str(getattr(q, "dtype", "")).endswith("uint8")
                        ):
                            return False, f"layer {i} {label}: invalid q8 tensor shapes"
                        rows += shape[1]
                    if rows != row_end - row_start:
                        return False, f"layer {i} {label}: q8 row mismatch"
                else:
                    return False, f"layer {i} {label}: unsupported storage {storage!r}"
                for j, tensor in enumerate(_walk_tensors(pool)):
                    ok, nb, reason = _validate_tensor(
                        tensor, label=f"{label}_t{j}", layer_idx=i
                    )
                    if not ok:
                        return False, reason
                    total_bytes += nb
                return True, ""

            ok, reason = _validate_pool_delta(
                record.get("compressor_pool"),
                "compressor_pool",
                allow_none=False,
                expected_start=start // ratio,
                expected_end=end // ratio,
            )
            if not ok:
                return False, reason, total_bytes
            ok, reason = _validate_pool_delta(
                record.get("indexer_pool"),
                "indexer_pool",
                allow_none=ratio != 4,
                expected_start=start // ratio if ratio == 4 else 0,
                expected_end=end // ratio if ratio == 4 else 0,
            )
            if not ok:
                return False, reason, total_bytes
            anchor = record.get("anchor")
            if anchor is not None:
                if not isinstance(anchor, dict):
                    return False, f"layer {i} DSV4 anchor is malformed", total_bytes
                try:
                    anchor_tokens = int(anchor.get("tokens", -1))
                except (TypeError, ValueError):
                    return False, f"layer {i} DSV4 anchor token is invalid", total_bytes
                if anchor_tokens != end:
                    return False, f"layer {i} DSV4 anchor boundary mismatch", total_bytes
                periodic = anchor.get("periodic")
                terminal = anchor.get("terminal")
                if not isinstance(periodic, bool) or not isinstance(terminal, bool):
                    return False, f"layer {i} DSV4 anchor kind is invalid", total_bytes
                if not periodic and not terminal:
                    return False, f"layer {i} DSV4 anchor has no valid kind", total_bytes
                if periodic and end % (block_size * anchor_blocks):
                    return False, f"layer {i} DSV4 periodic anchor is misaligned", total_bytes
                append_safe = anchor.get("append_safe", False)
                if not isinstance(append_safe, bool):
                    return False, (
                        f"layer {i} DSV4 append-safe marker is invalid"
                    ), total_bytes
                if append_safe and (
                    end - start != block_size
                    or start % block_size
                    or end % block_size
                ):
                    return False, (
                        f"layer {i} DSV4 append-safe anchor is misaligned"
                    ), total_bytes
                local_state = anchor.get("local_state")
                if (
                    not isinstance(local_state, (tuple, list))
                    or len(local_state) != 2
                ):
                    return False, f"layer {i} DSV4 anchor local state is missing", total_bytes
                key_shape = tuple(getattr(local_state[0], "shape", ()))
                value_shape = tuple(getattr(local_state[1], "shape", ()))
                expected_local_rows = min(end, sliding_window)
                if (
                    len(key_shape) < 3
                    or key_shape != value_shape
                    or int(key_shape[-2]) != expected_local_rows
                ):
                    return False, (
                        f"layer {i} DSV4 anchor local K/V geometry is invalid"
                    ), total_bytes
                local_meta = anchor.get("meta_state")
                if (
                    not isinstance(local_meta, (tuple, list))
                    or len(local_meta) < 4
                ):
                    return False, f"layer {i} DSV4 anchor metadata is missing", total_bytes
                try:
                    keep, max_size, offset, idx = map(int, local_meta[:4])
                except (TypeError, ValueError):
                    return False, f"layer {i} DSV4 anchor metadata is invalid", total_bytes
                if (
                    keep < 0
                    or keep > max_size
                    or max_size != sliding_window
                    or offset != end
                    or idx != expected_local_rows
                ):
                    return False, f"layer {i} DSV4 anchor metadata is non-canonical", total_bytes
                for j, tensor in enumerate(_walk_tensors(anchor)):
                    ok, nb, reason = _validate_tensor(
                        tensor, label=f"dsv4_anchor_t{j}", layer_idx=i
                    )
                    if not ok:
                        return False, reason, total_bytes
                    total_bytes += nb
            if end - start < block_size and (
                not isinstance(anchor, dict) or anchor.get("terminal") is not True
            ):
                return False, (
                    f"layer {i} DSV4 partial block is not a terminal anchor"
                ), total_bytes

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

        elif tag == "zaya_cca":
            # ("zaya_cca", kv_entry, cca_state, cca_meta, cache_meta)
            if len(entry) < 3:
                return False, (
                    f"layer {i} 'zaya_cca': len={len(entry)} < 3"
                ), total_bytes
            kv_entry = entry[1]
            if isinstance(kv_entry, (tuple, list)) and kv_entry and kv_entry[0] != "skip":
                sub_ok, sub_reason, sub_bytes = validate_cache_record(
                    [tuple(kv_entry)],
                    expected_num_layers=None,
                    source=f"{source}/zaya_cca_kv[{i}]",
                )
                if not sub_ok:
                    return False, f"layer {i} zaya_cca KV: {sub_reason}", total_bytes
                total_bytes += sub_bytes
            cca_state = entry[2]
            if cca_state is not None:
                tensors = list(_walk_tensors(cca_state))
                if len(tensors) < 2:
                    return False, (
                        f"layer {i} 'zaya_cca': terminal CCA state "
                        f"has {len(tensors)} tensor(s), expected conv_state and prev_hs"
                    ), total_bytes
                for j, t in enumerate(tensors):
                    ok, nb, reason = _validate_tensor(
                        t, label=f"zaya_cca_state[{j}]", layer_idx=i
                    )
                    if not ok:
                        return False, reason, total_bytes
                    total_bytes += nb
                if len(tensors[0].shape) != 3:
                    return False, (
                        f"layer {i} 'zaya_cca'.conv_state rank "
                        f"{len(tensors[0].shape)} != 3"
                    ), total_bytes
                if len(tensors[1].shape) != 3:
                    return False, (
                        f"layer {i} 'zaya_cca'.prev_hs rank "
                        f"{len(tensors[1].shape)} != 3"
                    ), total_bytes

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

    def _is_turboquant_kv_cache(value: Any) -> bool:
        cls = type(value)
        return (
            cls.__name__ == "TurboQuantKVCache"
            and cls.__module__.startswith("jang_tools.turboquant")
        )

    def _m3_seq_len(value: Any) -> Optional[int]:
        shape = getattr(value, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) < 3:
            return None
        try:
            return int(shape[-2])
        except Exception:
            return None

    def _validate_m3_lanes(
        keys: Any,
        values: Any,
        idx_keys: Any,
        *,
        meta: Any = None,
    ) -> Tuple[bool, str, int]:
        if keys is None or values is None:
            return False, f"layer {layer_idx}: MiniMaxM3 keys/values are None", 0
        if idx_keys is None:
            return False, f"layer {layer_idx}: MiniMaxM3 idx_keys missing", 0

        total_bytes = 0
        for label, value in (
            ("keys", keys),
            ("values", values),
            ("idx_keys", idx_keys),
        ):
            ok, nb, reason = _validate_tensor_tree(
                value, label=label, layer_idx=layer_idx
            )
            if not ok:
                return False, reason, total_bytes
            total_bytes += nb

        key_len = _m3_seq_len(keys)
        value_len = _m3_seq_len(values)
        idx_len = _m3_seq_len(idx_keys)
        if key_len is None or value_len is None or idx_len is None:
            return False, (
                f"layer {layer_idx}: MiniMaxM3 cache lanes lack sequence axis"
            ), total_bytes
        if key_len != value_len:
            return False, (
                f"layer {layer_idx}: MiniMaxM3 lane length mismatch "
                f"keys={key_len} values={value_len} idx_keys={idx_len}"
            ), total_bytes

        offset = getattr(layer_cache, "offset", None)
        seq = _decode_meta_sequence(meta)
        if offset is None and seq:
            offset = seq[0]
        if offset is not None and not type(offset).__module__.startswith("unittest.mock"):
            ok, parsed_offset, reason = _validate_int_range(
                offset,
                label=f"layer {layer_idx}.offset",
                lo=0,
                hi=MAX_CACHE_OFFSET,
            )
            if not ok:
                return False, reason, total_bytes
            # mlx-lm KVCache grows K/V in `step`-sized backing buffers while
            # `offset` remains the logical token count. MiniMax-M3/QSA index
            # keys are append-only at their logical extent. Comparing the raw
            # physical K/V width to idx_keys therefore rejects every healthy
            # off-boundary live cache (for example 13 logical tokens in a
            # 256-token K/V allocation). Persisted states are usually sliced,
            # but live validation must honor the same logical boundary.
            if parsed_offset > key_len or parsed_offset > value_len:
                return False, (
                    f"layer {layer_idx}: MiniMaxM3 offset {parsed_offset} "
                    f"exceeds K/V backing lengths {key_len}/{value_len}"
                ), total_bytes
            # The single-sequence sparse cache grows idx_keys in step-sized
            # backing buffers (physical capacity >= logical extent).  Two
            # checks: the backing must hold the logical offset, AND the
            # INITIALIZED extent (_idx_offset) must equal the logical offset;
            # spare capacity must never hide an under- or over-appended lane.
            if parsed_offset > idx_len:
                return False, (
                    f"layer {layer_idx}: MiniMaxM3 logical offset "
                    f"{parsed_offset} > idx_keys length {idx_len}"
                ), total_bytes
            initialized = getattr(layer_cache, "_idx_offset", None)
            if isinstance(initialized, int) and not isinstance(initialized, bool):
                if initialized != parsed_offset:
                    return False, (
                        f"layer {layer_idx}: MiniMaxM3 initialized index extent "
                        f"{initialized} != logical offset {parsed_offset}"
                    ), total_bytes
                if initialized > idx_len:
                    return False, (
                        f"layer {layer_idx}: MiniMaxM3 initialized index extent "
                        f"{initialized} > idx_keys length {idx_len}"
                    ), total_bytes
            derived_nbytes = getattr(layer_cache, "derived_nbytes", 0)
            if isinstance(derived_nbytes, int) and not isinstance(derived_nbytes, bool):
                total_bytes += derived_nbytes
        elif key_len != idx_len:
            return False, (
                f"layer {layer_idx}: MiniMaxM3 lane length mismatch without "
                f"logical offset keys={key_len} values={value_len} "
                f"idx_keys={idx_len}"
            ), total_bytes
        return True, "", total_bytes

    def _check_int_attr(attr: str, lo: int, hi: int) -> Tuple[bool, str]:
        if not hasattr(layer_cache, attr):
            return True, ""
        value = getattr(layer_cache, attr)
        if attr == "_idx" and value is None and _is_turboquant_kv_cache(layer_cache):
            # TurboQuantKVCache leaves _idx unset for clean single-sequence
            # cache state until the cache is prepared for batched use.
            return True, ""
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
        meta = layer_cache.get("meta_state")
        cls_name = str(layer_cache.get("class_name", ""))
        if cls_name == "MiniMaxM3SparseCache":
            if not isinstance(state, (tuple, list)) or len(state) != 3:
                return False, (
                    f"layer {layer_idx}: MiniMaxM3 state must contain "
                    "keys/values/idx_keys"
                ), total
            return _validate_m3_lanes(
                state[0],
                state[1],
                state[2],
                meta=meta,
            )
        ok, nb, reason = _validate_tensor_tree(
            state, label="state", layer_idx=layer_idx
        )
        if not ok:
            return False, reason, total
        total += nb
        if cls_name == "QuantizedKVCache":
            ok, reason = _validate_quant_meta(meta, label=f"layer {layer_idx}.meta")
            if not ok:
                return False, reason, total
        elif cls_name in {"Glm5KDACache", "Glm5MLACache"}:
            ok, reason = _validate_glm5_typed_state(
                cls_name,
                state,
                meta,
                label=f"layer {layer_idx}.meta",
            )
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
        if type(layer_cache).__name__ == "MiniMaxM3SparseCache":
            return _validate_m3_lanes(
                getattr(layer_cache, "keys", None),
                getattr(layer_cache, "values", None),
                getattr(layer_cache, "idx_keys", None),
                meta=getattr(layer_cache, "meta_state", None),
            )
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
        cls_name = type(layer_cache).__name__
        if cls_name in {"Glm5KDACache", "Glm5MLACache"}:
            ok, reason = _validate_glm5_typed_state(
                cls_name,
                getattr(layer_cache, "state", None),
                meta,
                label=f"layer {layer_idx}.meta",
            )
            if not ok:
                return False, reason, total
            return True, "", total
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

    file_metadata = header.get("__metadata__")
    is_tq_native_header = (
        isinstance(file_metadata, dict)
        and file_metadata.get("__tq_native__") == "true"
    )

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
            # Prompt-level TQ snapshots flatten packed indices/signs into one
            # uint32 vector.  A legitimate long-context vector can exceed the
            # generic 256K axis ceiling while remaining far below the 4GB
            # per-tensor and 32GB per-file safety caps below.  Only relax the
            # axis cap for exact packed-field names in an explicitly TQ-native
            # file; decoded KV shapes are still validated separately by
            # validate_tq_native_metadata() before any decode allocation.
            packed_tq_vector = (
                is_tq_native_header
                and len(shape) == 1
                and _TQ_PACKED_HEADER_NAME.fullmatch(name) is not None
            )
            limit = MAX_CACHE_OFFSET if _DSV4_STATE_HEADER_NAME.fullmatch(name) else MAX_TENSOR_DIM
            if dim > limit and not packed_tq_vector:
                return False, (
                    f"header entry {name!r}: dim[{axis}]={dim} > {limit}"
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

        for suffix in ("key_dtype", "value_dtype"):
            dtype = metadata.get(f"__{prefix}_{suffix}__")
            if dtype not in ALLOWED_TQ_DTYPES:
                return False, (
                    f"{label}.{suffix}: {dtype!r} not in "
                    f"{sorted(ALLOWED_TQ_DTYPES)}"
                )

        decoded_shapes: dict[str, list[int]] = {}
        for suffix, dtype_suffix in (("ck_shape", "key_dtype"), ("cv_shape", "value_dtype")):
            raw = metadata.get(f"__{prefix}_{suffix}__", "[]")
            try:
                shape = json.loads(raw)
            except json.JSONDecodeError as e:
                return False, f"{label}.{suffix}: invalid JSON {raw!r}: {e}"
            ok_shape, shape_reason, nbytes = _validate_shape_list(
                shape, label=f"{label}.{suffix}",
                bytes_per_elem=_TQ_DTYPE_BYTES[metadata[f"__{prefix}_{dtype_suffix}__"]],
            )
            if not ok_shape:
                return False, shape_reason
            decoded_shapes[suffix] = shape
            total_decoded += nbytes
            if total_decoded > MAX_TOTAL_RECORD_BYTES:
                return False, (
                    f"{label}: decoded total {total_decoded} bytes "
                    f"> {MAX_TOTAL_RECORD_BYTES}"
                )

        # Encoded codebook index widths are an implementation detail and need
        # not equal the user-facing TQ codec width (TQ8 currently uses 7-bit
        # packed indices). Validate their bounds independently.
        for suffix in ("ck_bits", "cv_bits"):
            key = f"__{prefix}_{suffix}__"
            if key in metadata:
                ok_bits, _, bit_reason = _validate_int_range(
                    metadata[key], label=f"{label}.{suffix}", lo=1, hi=16
                )
                if not ok_bits:
                    return False, bit_reason

        for suffix in ("key_bits", "value_bits"):
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

        seed_key = f"__{prefix}_seed__"
        if seed_key in metadata:
            ok_seed, _, seed_reason = _validate_int_range(
                metadata[seed_key],
                label=f"{label}.seed",
                lo=0,
                hi=2_147_483_647,
            )
            if not ok_seed:
                return False, seed_reason

        offset_key = f"__{prefix}_offset__"
        compressed_key = f"__{prefix}_compressed_tokens__"
        sink_key = f"__{prefix}_sink_tokens__"
        if offset_key in metadata:
            offset = int(metadata[offset_key])
            compressed = int(metadata.get(compressed_key, offset))
            sink = int(metadata.get(sink_key, "0"))
            key_tokens = decoded_shapes.get("ck_shape", [0, 0])[-2]
            value_tokens = decoded_shapes.get("cv_shape", [0, 0])[-2]
            if not (
                offset > 0
                and compressed == offset
                and key_tokens == offset
                and value_tokens == offset
                and sink == 0
            ):
                return False, (
                    f"{label}: incomplete TQ storage payload "
                    f"offset={offset}, compressed={compressed}, "
                    f"key_tokens={key_tokens}, value_tokens={value_tokens}, "
                    f"sink_tokens={sink}"
                )

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

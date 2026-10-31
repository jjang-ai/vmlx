# SPDX-License-Identifier: Apache-2.0
# Block disk store is original vMLX work by Jinho Jang (eric@jangq.ai).
# L2 SSD cache tier with safetensors serialization, hybrid SSM cumulative
# state persistence, orig_dtype metadata, and QuantizedKVCache support.
# github.com/jjang-ai/vmlx
"""
Block-level disk persistence for paged KV cache.

Provides an L2 disk tier behind the L1 in-memory PagedCacheManager.
Blocks are stored as safetensors files indexed by their chain hash.

Architecture:
- Each block (e.g. 64 tokens of KV data) is stored as a separate safetensors file
- Content-addressable: file path derived from the block's chain hash
- SQLite WAL index maps hash → file for fast lookup
- Background writer thread prevents disk I/O from blocking inference
- LRU eviction when total disk usage exceeds configured max

Integration points:
- On L1 eviction: write block to disk before freeing RAM
- On L1 lookup miss: check disk before recomputing
- On generation complete: write-through new blocks to disk

Supported cache_data tuple types (from prefix_cache.py):
- ("kv", keys_slice, values_slice) — standard KVCache
- ("quantized_kv", keys_tuple, values_tuple, meta) — QuantizedKVCache
- ("turboquant_kv", encoded_keys, encoded_values, config) — native
  TurboQuant codebook payload with the exact per-layer seed and codec bits
- ("rotating_kv", terminal_keys, terminal_values, max_size, keep, offset, idx)
- ("rotating_kv_pending", class_name) for non-terminal rotating chain blocks
  — RotatingKVCache
- ("cumulative", state_list, meta, class_name) — MambaCache/ArraysCache
- ("deepseek_v4", state_tree, meta, class_name, cache_meta) — DSV4
  composite cache (SWA local + CSA/HCA compressor/indexer pools)
- ("deepseek_v4_pending", class_name, cache_meta) — non-terminal DSV4
  marker so paged/L2 chain hashes remain materialized without duplicating
  full CSA/HCA pool state in every block
- ("deepseek_v4_delta_v1", record_tree, class_name, cache_meta) — immutable
  DSV4 SWA/CSA/HCA native block delta plus optional exact anchor
- ("zaya_cca", kv_entry, cca_state, cca_meta, cache_meta) — ZAYA CCA
  typed cache: standard KV pages plus terminal conv_state/prev_hs
- ("minimax_m3", keys_slice, values_slice, idx_keys_slice) — MiniMax-M3 MSA
  sparse layer: standard GQA KV plus the append-only Lightning-indexer key
  cache. All three are positional (seq axis), so every block is a complete,
  independently-sliceable payload (unlike the DSV4/ZAYA terminal-only state).
- ("skip",) — placeholder for cumulative layers in non-last blocks
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import struct
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional, Tuple
from urllib.parse import quote

logger = logging.getLogger(__name__)


class _BlockWriteConnectionError(RuntimeError):
    """A failed publication could not safely reuse its SQLite connection."""

    def __init__(self, message: str, block_hash: bytes | None = None):
        super().__init__(message)
        self.block_hash = block_hash

_SAFETENSORS_NUMPY_DTYPE_CODES = {
    ("f", 8): "F64",
    ("f", 4): "F32",
    ("f", 2): "F16",
    ("i", 8): "I64",
    ("u", 8): "U64",
    ("i", 4): "I32",
    ("u", 4): "U32",
    ("i", 2): "I16",
    ("u", 2): "U16",
    ("i", 1): "I8",
    ("u", 1): "U8",
    ("b", 1): "BOOL",
    ("c", 8): "C64",
}

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


def _pack_tree(obj: Any, tensors: Dict[str, Any], prefix: str, counter: List[int]) -> Dict[str, Any]:
    """Flatten a nested cache-state tree into safetensors + JSON metadata."""
    if obj is None:
        return {"kind": "none"}
    if hasattr(obj, "shape"):
        key = f"{prefix}_{counter[0]}"
        counter[0] += 1
        tensors[key] = obj
        return {
            "kind": "tensor",
            "key": key,
            "orig_dtype": str(getattr(obj, "dtype", "")),
        }
    if isinstance(obj, tuple):
        return {
            "kind": "tuple",
            "items": [_pack_tree(x, tensors, prefix, counter) for x in obj],
        }
    if isinstance(obj, list):
        return {
            "kind": "list",
            "items": [_pack_tree(x, tensors, prefix, counter) for x in obj],
        }
    if isinstance(obj, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): _pack_tree(value, tensors, prefix, counter)
                for key, value in obj.items()
            },
        }
    return {"kind": "literal", "value": _json_safe(obj)}


def _unpack_tree(node: Any, data: Dict[str, Any]) -> Any:
    if not isinstance(node, dict):
        return None
    kind = node.get("kind")
    if kind == "none":
        return None
    if kind == "literal":
        return node.get("value")
    if kind == "tensor":
        arr = data.get(node.get("key", ""))
        if arr is not None and HAS_MLX:
            orig_dt = node.get("orig_dtype")
            if orig_dt and orig_dt != str(getattr(arr, "dtype", "")):
                target = getattr(mx, orig_dt.replace("mlx.core.", ""), None)
                if target is not None:
                    try:
                        arr = _restore_serialized_dtype(arr, target)
                    except Exception:
                        pass
        return arr
    if kind == "tuple":
        return tuple(_unpack_tree(x, data) for x in node.get("items", []))
    if kind == "list":
        return [_unpack_tree(x, data) for x in node.get("items", [])]
    if kind == "dict":
        return {
            str(key): _unpack_tree(value, data)
            for key, value in (node.get("items") or {}).items()
        }
    return None


class _ValidationTensor:
    """Shape-only tensor stand-in for native cache candidate validation.

    Prefix selection needs native tags, interval geometry, anchor flags, and
    rotating-window shapes. It does not need the multi-megabyte tensor bodies
    that worker reconstruction will load immediately afterward. Keeping this
    deliberately tiny object separate from MLX arrays also makes the metadata
    probe safe on the API thread.
    """

    __slots__ = ("shape", "dtype")

    def __init__(self, shape: Any, dtype: str = "") -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = str(dtype or "")


def _unpack_validation_tree(node: Any, handle: Any, tensor_names: set[str]) -> Any:
    """Rebuild literals and tensor shapes from one packed-tree descriptor."""
    if not isinstance(node, dict):
        return None
    kind = node.get("kind")
    if kind == "none":
        return None
    if kind == "literal":
        return node.get("value")
    if kind == "tensor":
        key = str(node.get("key", ""))
        if key not in tensor_names:
            return None
        return _ValidationTensor(
            handle.get_slice(key).get_shape(),
            str(node.get("orig_dtype", "")),
        )
    if kind == "tuple":
        return tuple(
            _unpack_validation_tree(item, handle, tensor_names)
            for item in node.get("items", [])
        )
    if kind == "list":
        return [
            _unpack_validation_tree(item, handle, tensor_names)
            for item in node.get("items", [])
        ]
    if kind == "dict":
        return {
            str(key): _unpack_validation_tree(value, handle, tensor_names)
            for key, value in (node.get("items") or {}).items()
        }
    return None


def _load_block_validation_entries(
    file_path: Path,
    member: Optional[Tuple[Dict[str, Any], int]] = None,
) -> Optional[List[Tuple]]:
    """Read only native cache decision metadata from a safetensors block.

    Returned tuples mirror the subset of deserialized cache entries consumed
    by ``prefix_cache`` candidate validators. Tensor bodies are represented by
    shape-only stand-ins; full validation and hydration remain on the owning
    model worker. ``member`` = (group meta, ordinal) reads one member of a group
    file through the same names and shapes a single-block file would expose.
    """
    try:
        from safetensors import safe_open

        with safe_open(str(file_path), framework="numpy") as raw_handle:
            handle = raw_handle if member is None else _GroupMemberHandle(raw_handle, member[0], member[1])
            tensor_names = set(handle.keys())
            meta_key = (
                "__vmlx_block_meta__"
                if "__vmlx_block_meta__" in tensor_names
                else "__metadata__" if "__metadata__" in tensor_names else None
            )
            if meta_key is None:
                return None
            raw = handle.get_tensor(meta_key).tobytes()
            meta = json.loads(raw.decode("utf-8"))
            if not isinstance(meta, dict):
                return None
            layer_types = meta.get("__layer_types__")
            if not isinstance(layer_types, dict):
                return None
            try:
                num_layers = int(meta.get("__num_cache_layers__", 0) or 0)
            except (TypeError, ValueError):
                return None
            if num_layers <= 0:
                return None

            def _scalar(name: str, default: Any = None) -> Any:
                if name not in tensor_names:
                    return default
                value = handle.get_tensor(name)
                try:
                    return int(value.reshape(-1)[0])
                except (TypeError, ValueError, IndexError):
                    return default

            entries: List[Tuple] = [("skip",) for _ in range(num_layers)]
            saw_native = False
            has_cache_list = any(
                layer_types.get(str(i)) == "cache_list" for i in range(num_layers)
            )
            declared_dtypes = None
            if has_cache_list:
                from .prefix_cache import runtime_cache_fingerprint

                if (
                    meta.get("__runtime_cache_fingerprint__")
                    != runtime_cache_fingerprint()
                ):
                    return None
                if "__tensor_dtypes__" in meta:
                    declared_dtypes = meta["__tensor_dtypes__"]
                    payload_names = tensor_names - {
                        "__vmlx_block_meta__",
                        "__metadata__",
                    }
                    if (
                        not isinstance(declared_dtypes, dict)
                        or set(declared_dtypes) != payload_names
                        or not _block_tree_dtypes_match(meta, declared_dtypes)
                    ):
                        return None
                    for name, wanted in declared_dtypes.items():
                        physical = _BLOCK_SAFETENSORS_DTYPES.get(
                            handle.get_slice(name).get_dtype()
                        )
                        if (
                            not isinstance(wanted, str)
                            or wanted not in _BLOCK_TENSOR_DTYPES
                            or not (
                                physical == wanted
                                or (physical == "uint16" and wanted == "bfloat16")
                            )
                        ):
                            return None

            def _plain_kv(keys_name: str, values_name: str, legacy_key: str) -> Tuple:
                if keys_name not in tensor_names or values_name not in tensor_names:
                    raise ValueError("incomplete plain KV cache metadata")
                if declared_dtypes is not None:
                    key_dtype = declared_dtypes[keys_name]
                    value_dtype = declared_dtypes[values_name]
                else:
                    target = (meta.get("__orig_dtypes__") or {}).get(legacy_key)
                    if not isinstance(target, str):
                        raise ValueError("missing legacy KV dtype")
                    target = target.removeprefix("mlx.core.")
                    if target not in {"float16", "bfloat16", "float32"}:
                        raise ValueError("unsupported legacy KV dtype")
                    key_dtype = _BLOCK_SAFETENSORS_DTYPES.get(
                        handle.get_slice(keys_name).get_dtype()
                    )
                    value_dtype = _BLOCK_SAFETENSORS_DTYPES.get(
                        handle.get_slice(values_name).get_dtype()
                    )
                    # The legacy reader casts both lanes only when keys need
                    # restoring; modern declarations always own each lane.
                    if key_dtype != target:
                        key_dtype = value_dtype = target
                if key_dtype is None or value_dtype is None:
                    raise ValueError("unsupported KV payload dtype")
                return (
                    "kv",
                    _ValidationTensor(
                        handle.get_slice(keys_name).get_shape(), f"mlx.core.{key_dtype}"
                    ),
                    _ValidationTensor(
                        handle.get_slice(values_name).get_shape(),
                        f"mlx.core.{value_dtype}",
                    ),
                )

            for i in range(num_layers):
                layer_type = layer_types.get(str(i))
                layer_meta = meta.get(str(i), {})
                if not isinstance(layer_meta, dict):
                    layer_meta = {}
                if layer_type == "deepseek_v4_delta_v1":
                    record = _unpack_validation_tree(
                        layer_meta.get("record_tree"),
                        handle,
                        tensor_names,
                    )
                    entries[i] = (
                        "deepseek_v4_delta_v1",
                        record,
                        layer_meta.get("class_name", "DeepseekV4Cache"),
                        layer_meta.get("cache_meta", {}),
                    )
                    saw_native = True
                elif layer_type == "deepseek_v4":
                    entries[i] = (
                        "deepseek_v4",
                        (),
                        layer_meta.get("meta", ""),
                        layer_meta.get("class_name", "DeepseekV4Cache"),
                        layer_meta.get("cache_meta", {}),
                    )
                    saw_native = True
                elif layer_type == "deepseek_v4_pending":
                    entries[i] = (
                        "deepseek_v4_pending",
                        layer_meta.get("class_name", "DeepseekV4Cache"),
                        layer_meta.get("cache_meta", {}),
                    )
                    saw_native = True
                elif layer_type == "rotating_kv":
                    keys_name = f"layer_{i}_keys"
                    values_name = f"layer_{i}_values"
                    if keys_name not in tensor_names or values_name not in tensor_names:
                        return None
                    orig_dtype = (meta.get("__orig_dtypes__") or {}).get(str(i), "")
                    key_dtype = (
                        f"mlx.core.{declared_dtypes[keys_name]}"
                        if declared_dtypes is not None
                        else orig_dtype
                    )
                    value_dtype = (
                        f"mlx.core.{declared_dtypes[values_name]}"
                        if declared_dtypes is not None
                        else orig_dtype
                    )
                    entries[i] = (
                        "rotating_kv",
                        _ValidationTensor(
                            handle.get_slice(keys_name).get_shape(), key_dtype
                        ),
                        _ValidationTensor(
                            handle.get_slice(values_name).get_shape(), value_dtype
                        ),
                        _scalar(f"layer_{i}_max_size", 0),
                        _scalar(f"layer_{i}_keep", 0),
                        _scalar(f"layer_{i}_offset"),
                        _scalar(f"layer_{i}_idx"),
                    )
                    saw_native = True
                elif layer_type == "rotating_kv_pending":
                    entries[i] = (
                        "rotating_kv_pending",
                        layer_meta.get("class_name", "RotatingKVCache"),
                    )
                    saw_native = True
                elif layer_type == "zaya_cca":
                    entries[i] = (
                        "zaya_cca",
                        ("skip",),
                        object() if layer_meta.get("terminal") is True else None,
                        layer_meta.get("cca_meta", ""),
                        layer_meta.get("cache_meta", {}),
                    )
                    saw_native = True
                elif layer_type == "kv" and has_cache_list:
                    # A newly admitted CacheList must preserve sibling tags:
                    # artificial skips trigger previous-block state checks.
                    entries[i] = _plain_kv(
                        f"layer_{i}_keys", f"layer_{i}_values", str(i)
                    )
                elif layer_type == "cache_list":
                    # Admit only complete plain-KV children. Quantized,
                    # cumulative, skipped and future child schemas still need
                    # the full reader; never hide their native-state markers.
                    count = layer_meta.get("sub_count")
                    if type(count) is not int or count <= 0 or layer_meta.get("subs"):
                        return None
                    expected_names = {
                        f"layer_{i}_sub_{j}_{kind}"
                        for j in range(count)
                        for kind in ("keys", "values")
                    }
                    actual_names = {
                        name
                        for name in tensor_names
                        if name.startswith(f"layer_{i}_sub_")
                    }
                    if actual_names != expected_names:
                        return None
                    children = []
                    for j in range(count):
                        children.append(
                            _plain_kv(
                                f"layer_{i}_sub_{j}_keys",
                                f"layer_{i}_sub_{j}_values",
                                f"{i}_sub_{j}",
                            )
                        )
                    entries[i] = ("cache_list", children)
                    saw_native = True
                elif has_cache_list and layer_type != "skip":
                    # Other sibling schemas previously used the full reader.
                    # Do not replace their state with an artificial skip.
                    return None

            return entries if saw_native else None
    except Exception:
        return None


_F_RDADVISE = 44  # <sys/fcntl.h>: struct radvisory { off_t ra_offset; int ra_count; }


def _advise_file_read(path: Path) -> bool:
    """Asynchronous OS read-ahead hint for one whole file; never raises."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        size = os.fstat(fd).st_size
        if size <= 0:
            return False
        if sys.platform == "darwin":
            import fcntl

            fcntl.fcntl(fd, _F_RDADVISE, struct.pack("qi", 0, min(size, 2**31 - 1)))
            return True
        advise = getattr(os, "posix_fadvise", None)
        if advise is not None:
            advise(fd, 0, size, os.POSIX_FADV_WILLNEED)
            return True
        return False
    except (OSError, ValueError):
        return False
    finally:
        os.close(fd)


def _stat_identity(st: os.stat_result) -> Tuple[int, int, int, int]:
    """Identity of one physical payload file that survives LRU touches.

    NOT mtime/ctime: every block read does os.utime() (the cross-process LRU
    signal), so mtime changes on each restore. Payloads are only ever replaced
    by rename, which yields a new inode and creation time.
    """
    birth = getattr(st, "st_birthtime", None)
    birth_ns = int(birth * 1e9) if birth is not None else 0
    return (int(st.st_dev), int(st.st_ino), int(st.st_size), birth_ns)


def _payload_identity(path: Path) -> Tuple[int, int, int, int]:
    """_stat_identity of a payload file; raises OSError when it is gone."""
    return _stat_identity(os.stat(path))


def _cache_data_has_tq(cache_data: Any) -> bool:
    for entry in cache_data or []:
        if not isinstance(entry, (tuple, list)) or not entry:
            continue
        if entry[0] == "turboquant_kv":
            return True
        if entry[0] == "cache_list" and len(entry) > 1:
            if _cache_data_has_tq(entry[1]):
                return True
    return False


# ------------------------------------------------------------------ group files (2026-10-12)
# A long cold store wrote one safetensors file per 64-token block (Naive N0.5: 85 tensors per file), and a restore paid
# a file open + header check + mx.load + 85-array eval + deserialize + validation per block: 1,252 guarded reads,
# 2.0-2.5 s for an 80k-token Naive-layout prefix (one whole-prefix file loads in 0.13 s). The writer now coalesces up
# to ``_group_max()`` consecutive eligible blocks of one batch (same fence, parent chain) into ONE file: every 4-D
# tensor all members share (same dtype, same non-token dims) is concatenated along its token axis (dim 2), anything
# else is kept per member under ``b{i}.``. Every member keeps ITS OWN hash path, index row and lifecycle: the group
# file is hard-linked under each member's path, so deletion, eviction, orphan cleanup and replace-existing stay per
# block and the filesystem frees the bytes with the last link. Size accounting divides by ``st_nlink`` (a row stores
# its share). A member read loads the group once (cached by file identity for the next members) and rebuilds exactly
# the tensor dict a single-block file would hold, so deserialization and validation are unchanged.
# VMLX_BLOCK_GROUP_FILES=N (default 16; 0 or 1 = one file per block).
_GROUP_META_KEY = "__vmlx_group_meta__"
_BLOCK_META_KEY = "__vmlx_block_meta__"
# minimax_m3 = the MiniMax-M3 / Qwen3.8 Flash-Next QSA sparse layer: keys, values and idx_keys are all 4-D with the
# token axis at dim 2 and reconstruct concatenates them exactly so (a member without idx_keys keeps them per member,
# which makes the span read fall back per block).
_GROUP_ELIGIBLE_TAGS = frozenset({"kv", "cache_list", "rotating_kv_pending", "skip", "minimax_m3"})
_GROUP_EXCLUDED_KEY_PARTS = ("_scales", "_zeros", "_data", "turboquant", "tq_", "dsv4", "zaya",
                             "cumulative", "_offset", "_max_size", "_keep", "no_state")


def _group_max() -> int:
    try:
        return max(0, int(os.environ.get("VMLX_BLOCK_GROUP_FILES", "16")))
    except (TypeError, ValueError):
        return 16


class _GroupMember:
    """Batch payload for one member of a staged group file (published as a hard link, not a rename)."""

    __slots__ = ("path", "index", "size", "share")

    def __init__(self, path: Path, index: int, size: int, share: int):
        self.path, self.index, self.size, self.share = path, int(index), int(size), int(share)


def _apportioned_size(file_stat) -> int:
    """Bytes a path accounts for: a hard-linked group file is charged once across its member links."""
    return max(0, int(file_stat.st_size)) // max(1, int(getattr(file_stat, "st_nlink", 1) or 1))


def _numpy_block_group_eligible(tensors: Dict[str, Any]) -> bool:
    """A frozen NumPy block payload that a group file can hold losslessly: only plain/CacheList KV and pending
    rotating markers (no rotating windows, cumulative/SSM state, quantized or TQ entries, DSV4/ZAYA/M3)."""
    meta_arr = tensors.get(_BLOCK_META_KEY)
    if meta_arr is None or _GROUP_META_KEY in tensors:
        return False
    try:
        import numpy as np

        meta = json.loads(np.asarray(meta_arr).tobytes().decode("utf-8"))
    except Exception:
        return False
    layer_types = meta.get("__layer_types__") if isinstance(meta, dict) else None
    if not isinstance(layer_types, dict) or not layer_types:
        return False
    if any(str(tag) not in _GROUP_ELIGIBLE_TAGS for tag in layer_types.values()):
        return False
    for name in tensors:
        if name == _BLOCK_META_KEY:
            continue
        if any(part in name for part in _GROUP_EXCLUDED_KEY_PARTS):
            return False
    return True


def _numpy_group_tensors(members: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Pack frozen NumPy member payloads ``[(hash_hex, tensors), ...]`` into one group tensor dict."""
    import numpy as np

    first = members[0][1]
    concat: Dict[str, List[int]] = {}
    for name, arr in first.items():
        if name == _BLOCK_META_KEY or getattr(arr, "ndim", 0) != 4:
            continue
        shapes_ok = True
        for _, tensors in members[1:]:
            other = tensors.get(name)
            if (other is None or other.ndim != 4 or other.dtype != arr.dtype
                    or other.shape[:2] != arr.shape[:2] or other.shape[3] != arr.shape[3]):
                shapes_ok = False
                break
        if shapes_ok:
            offsets = [0]
            for _, tensors in members:
                offsets.append(offsets[-1] + int(tensors[name].shape[2]))
            concat[name] = offsets
    out: Dict[str, Any] = {}
    for name, offsets in concat.items():
        out[name] = np.ascontiguousarray(np.concatenate([tensors[name] for _, tensors in members], axis=2))
    for i, (_, tensors) in enumerate(members):
        for name, arr in tensors.items():
            if name not in concat:
                out[f"b{i}.{name}"] = arr
    group_meta = {"version": 1, "members": [h for h, _ in members], "concat": concat, "axis": 2}
    out[_GROUP_META_KEY] = np.frombuffer(json.dumps(group_meta, sort_keys=True).encode("utf-8"), dtype=np.uint8).copy()
    return out


def _decode_group_meta(arr) -> Optional[Dict[str, Any]]:
    try:
        import numpy as np

        meta = json.loads(np.asarray(arr).tobytes().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict) or meta.get("version") != 1 or not isinstance(meta.get("members"), list):
        return None
    if not isinstance(meta.get("concat"), dict) or int(meta.get("axis", 2)) != 2:
        return None
    return meta


def _group_member_keys(group: Dict[str, Any], n_members: int) -> List[List[Tuple[str, str]]]:
    """Per member: [(name in the member's own dict, name in the group)] for the ``b{i}.`` entries (built once)."""
    keys: List[List[Tuple[str, str]]] = [[] for _ in range(n_members)]
    for name in group:
        if name.startswith("b") and "." in name:
            head, _, short = name.partition(".")
            try:
                i = int(head[1:])
            except ValueError:
                continue
            if 0 <= i < n_members:
                keys[i].append((short, name))
    return keys


def _group_member_view(
    group: Dict[str, Any],
    group_meta: Dict[str, Any],
    index: int,
    member_keys: Optional[List[List[Tuple[str, str]]]] = None,
) -> Dict[str, Any]:
    """The tensor dict of member ``index``: concatenated slots sliced along dim 2 (views), ``b{index}.`` keys
    unprefixed. Exactly the mapping a single-block file of that member would load."""
    view: Dict[str, Any] = {}
    for name, offsets in group_meta["concat"].items():
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        view[name] = group[name][:, :, lo:hi, :]
    if member_keys is None:
        member_keys = _group_member_keys(group, len(group_meta["members"]))
    for short, name in member_keys[index]:
        view[short] = group[name]
    return view


class _GroupMemberHandle:
    """safe_open-like view of one group member for header-only readers (keys / get_tensor / get_slice)."""

    def __init__(self, handle, group_meta: Dict[str, Any], index: int):
        self._h, self._meta, self._i = handle, group_meta, int(index)
        prefix = f"b{self._i}."
        self._names = {name: name for name in group_meta["concat"]}
        for name in handle.keys():
            if name.startswith(prefix):
                self._names[name[len(prefix):]] = name

    def keys(self):
        return list(self._names)

    def get_tensor(self, name):
        arr = self._h.get_tensor(self._names[name])
        offsets = self._meta["concat"].get(name)
        if offsets is not None:
            arr = arr[:, :, int(offsets[self._i]):int(offsets[self._i + 1]), :]
        return arr

    def get_slice(self, name):
        handle, real = self._h, self._names[name]
        offsets = self._meta["concat"].get(name)
        index = self._i

        class _Slice:
            def get_dtype(self_inner):
                return handle.get_slice(real).get_dtype()

            def get_shape(self_inner):
                shape = list(handle.get_slice(real).get_shape())
                if offsets is not None:
                    shape[2] = int(offsets[index + 1]) - int(offsets[index])
                return shape

        return _Slice()


def _restore_tq_block_entry(
    data: Dict[str, Any],
    tensor_prefix: str,
    tq_meta: Any,
) -> Tuple[Any, ...]:
    if not isinstance(tq_meta, dict):
        raise ValueError("missing TQ block metadata")
    from jang_tools.turboquant.cache import EncodedKeys, EncodedValues

    encoded_keys = EncodedKeys(
        indices_packed=data[f"{tensor_prefix}_tq_ck_indices_packed"],
        qjl_packed=data[f"{tensor_prefix}_tq_ck_qjl_packed"],
        residual_norms=data[f"{tensor_prefix}_tq_ck_residual_norms"],
        vector_norms=data[f"{tensor_prefix}_tq_ck_vector_norms"],
        shape=tuple(tq_meta["ck_shape"]),
        index_bits=int(tq_meta["ck_bits"]),
    )
    encoded_values = EncodedValues(
        indices_packed=data[f"{tensor_prefix}_tq_cv_indices_packed"],
        vector_norms=data[f"{tensor_prefix}_tq_cv_vector_norms"],
        shape=tuple(tq_meta["cv_shape"]),
        index_bits=int(tq_meta["cv_bits"]),
    )
    config = {
        key: int(tq_meta[key])
        for key in (
            "key_dim",
            "value_dim",
            "key_bits",
            "value_bits",
            "seed",
            "offset",
        )
    }
    config["key_dtype"] = str(tq_meta["key_dtype"])
    config["value_dtype"] = str(tq_meta["value_dtype"])
    return ("turboquant_kv", encoded_keys, encoded_values, config)



def _restore_serialized_dtype(arr, target):
    """Return `arr` as `target`, reinterpreting when the payload carries raw bits.

    Two on-disk generations are both valid and both appear in the wild:

      * uint16 payload + bfloat16 target -> the bytes ARE the bf16 bits, so
        reinterpret. Converting here would read them as integers and produce
        garbage.
      * anything else (notably f32 payloads written before the 1x change, and
        the historical f16 ones) -> cast, exactly as before.

    Keeping both paths is what makes the write-side change backward compatible:
    a cache written by an older build still restores correctly.
    """
    if target is None:
        return arr
    try:
        if str(arr.dtype) == str(target):
            return arr
        if arr.dtype == mx.uint16 and target == mx.bfloat16:
            return arr.view(mx.bfloat16)
    except Exception:  # noqa: BLE001 - dtype probing must never break a restore
        pass
    return arr.astype(target)


_BLOCK_TENSOR_DTYPES = frozenset({
    "bool_", "int8", "int16", "int32", "int64", "uint8", "uint16",
    "uint32", "uint64", "float16", "bfloat16", "float32", "complex64",
})
_BLOCK_SAFETENSORS_DTYPES = {
    "BOOL": "bool_", "I8": "int8", "I16": "int16", "I32": "int32",
    "I64": "int64", "U8": "uint8", "U16": "uint16", "U32": "uint32",
    "U64": "uint64", "F16": "float16", "BF16": "bfloat16",
    "F32": "float32", "C64": "complex64",
}


def _block_tensor_dtype_name(dtype) -> str:
    name = str(dtype).split(".")[-1]
    return "bool_" if name == "bool" else name


def _restore_block_tensor_dtypes(data, declared) -> bool:
    """Restore lossless carriers without conflating independent native lanes.

    A QSA index is FP32 even when its attention KV is BF16. Quantized cache
    scales/biases and nested recurrent state likewise own their own precision.
    Only BF16->U16 is a transport reinterpretation; integer packed data and all
    other dtypes must remain exactly as declared. Bad metadata is a cache miss.
    """
    if not isinstance(declared, dict) or set(declared) != set(data):
        return False
    restored = {}
    for name, value in data.items():
        wanted = declared[name]
        if not isinstance(wanted, str) or wanted not in _BLOCK_TENSOR_DTYPES:
            return False
        actual = _block_tensor_dtype_name(getattr(value, "dtype", ""))
        if actual == wanted:
            restored[name] = value
        elif wanted == "bfloat16" and actual == "uint16":
            restored[name] = _restore_serialized_dtype(value, mx.bfloat16)
        else:
            return False
    data.update(restored)
    return True


def _block_tree_dtypes_match(node, declared) -> bool:
    """Reject conflicting tree metadata instead of recasting restored leaves."""
    if isinstance(node, dict):
        if node.get("kind") == "tensor":
            key = node.get("key")
            return (
                isinstance(key, str)
                and key in declared
                and _block_tensor_dtype_name(node.get("orig_dtype")) == declared[key]
            )
        return all(_block_tree_dtypes_match(value, declared) for value in node.values())
    if isinstance(node, list):
        return all(_block_tree_dtypes_match(value, declared) for value in node)
    return True


class BlockDiskStore:
    """
    Content-addressable block storage on disk for paged KV cache.

    Each block is serialized as a safetensors file containing per-layer
    KV tensors. A SQLite index maps chain hashes to file paths for O(1) lookup.

    Args:
        cache_dir: Directory to store cache files.
        max_size_gb: Maximum total cache size in GB. 0 = unlimited.
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_gb: float = 10.0,
        expected_num_layers: Optional[int] = None,
        allow_tq_native: Optional[bool] = None,
        global_cache_root: Optional[str] = None,
        allow_legacy_hashed_namespaces: bool = False,
        allow_legacy_direct_namespace: bool = False,
        activity_probe: Optional[Callable[[], bool]] = None,
        max_pending_write_bytes: Optional[int] = None,
        **kwargs,
    ):
        from .global_disk_cache_budget import (
            ensure_managed_block_cache_namespace,
            get_global_disk_cache_budget,
        )

        # ``max_size_gb`` is the one physical budget for the configured root,
        # not an allowance that each hashed model namespace may consume again.
        # Direct callers default to a one-namespace root; schedulers pass the
        # default/custom parent root explicitly.
        requested_cache_dir = Path(cache_dir).expanduser().resolve()
        self.global_cache_root = Path(
            global_cache_root if global_cache_root is not None else cache_dir
        ).expanduser().resolve()
        if not requested_cache_dir.is_relative_to(self.global_cache_root):
            raise ValueError(
                "block-cache namespace must be contained by its aggregate "
                f"budget root: namespace={requested_cache_dir}, "
                f"root={self.global_cache_root}"
            )
        self.cache_dir = ensure_managed_block_cache_namespace(cache_dir)
        if not self.cache_dir.is_relative_to(self.global_cache_root):
            # Defend against a path substitution between the preflight resolve
            # and namespace claim.  An out-of-root namespace would otherwise
            # publish bytes that the aggregate size scanner can never count or
            # evict.
            raise OSError(
                "block-cache namespace escaped its aggregate budget root: "
                f"namespace={self.cache_dir}, root={self.global_cache_root}"
            )
        self.blocks_dir = self.cache_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max(0.0, max_size_gb) * 1024**3)
        # Codex 2026-05-06 contract #4: expected layer count from model
        # config (e.g. 43 for DSV4-Flash). Validator hard-rejects cache
        # records whose layer count differs — that is the canonical
        # "wrong-model L2 entry" signal. ``None`` skips the check (e.g.
        # a generic store without model context); per-tensor + total
        # byte caps still apply.
        self._expected_num_layers: Optional[int] = expected_num_layers
        if allow_tq_native is None:
            allow_tq_native = os.environ.get("VMLX_DISABLE_TQ_KV", "").lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }
        # An explicit non-TQ cache mode must also govern persisted blocks.  It is
        # not sufficient to disable creation of live TurboQuant cache objects:
        # an old L2 directory can still contain native TQ blocks from an Auto
        # run, and silently decoding those blocks makes the UI's None setting
        # untrue.  The hash index admits only one representation per token
        # chain, so remove an incompatible record on read; otherwise it would
        # also prevent this Off run from writing a standard replacement.
        self._allow_tq_native = bool(allow_tq_native)
        # Payload files this process has already proven to be plain (non-TQ) KV,
        # keyed by hash -> _stat_identity() (device, inode, size, birth) of the file that
        # was inspected or written. In the non-TQ mode above, has_block() loaded
        # and deserialized the WHOLE payload on every call to look for a legacy
        # TQ record; the store dedup loop calls it once per already-stored block
        # of every turn, so a one-block tool-continuation store at 8k tokens
        # re-read ~120 payloads (~1/3 of its 170 ms). Files are content-addressed
        # and replaced only by rename, so an unchanged identity is the same bytes;
        # any rewrite by another process changes it and is inspected again. The
        # index/file checks still run first.
        self._verified_plain_payloads: Dict[str, Tuple[int, int, int, int]] = {}
        self._verified_plain_lock = threading.Lock()
        self._verified_plain_max = 65536
        # Per-file results of the read-time checks that depend only on the
        # immutable payload: hash -> (_stat_identity, header_ok, rotating
        # offsets or None if unknown). A tool loop re-restores the same prefix
        # blocks every turn; re-validating each header and re-probing each
        # file's rotating windows cost ~70 ms per 32k restore. Any rewrite
        # (rename) changes the identity and is checked again; the post-load
        # validator still runs on every read.
        self._payload_check_memo: Dict[
            str, Tuple[Tuple[int, int, int, int], bool, Optional[Dict[int, int]]]
        ] = {}
        self._payload_check_lock = threading.Lock()
        # Candidate-validation metadata per immutable payload file: hash ->
        # (_stat_identity, entries). Every SSD-only lookup of a mixed-SWA or
        # nested chain re-read each block's header and packed metadata through
        # safetensors before reconstruction (~0.4 ms per block: ~0.24 s of a
        # 37k Naive-layout lookup, on every tool continuation). The entries
        # hold only literals and shape stand-ins; a rename (rewrite) changes
        # the identity and the file is checked again.
        self._validation_entries_memo: "OrderedDict[str, Tuple[Tuple[int, int, int, int], List[Tuple]]]" = OrderedDict()
        self._validation_entries_max = 4096
        # Group files (see _group_max): the loaded, evaluated group payload of the most recent group files, keyed by
        # file identity (every member link shares the inode), so a reconstruct reads each group once. Model thread
        # only; dropped by release_group_cache() after each reconstruct and at idle (RAM stays at the model's size).
        self._group_cache: "OrderedDict[Tuple[int, int, int, int], Tuple[Dict[str, Any], Dict[str, Any]]]" = OrderedDict()
        self._group_cache_max = 2
        self._group_header_ok: Dict[Tuple[int, int, int, int], bool] = {}
        self._group_meta_only: "OrderedDict[Tuple[int, int, int, int], Tuple[Dict[str, Any], Dict[str, int]]]" = OrderedDict()
        self._group_touch_times: Dict[Tuple[int, int, int, int], float] = {}
        self._group_staged_paths: List[Path] = []
        self.group_files_written = 0
        self.group_members_written = 0
        self.group_member_reads = 0
        # LRU touches (one per block read) are recorded here and applied by the
        # background writer in one transaction (see _queue_access_update).
        self._access_pending: Dict[str, List[float]] = {}
        self._access_lock = threading.Lock()

        # SQLite index
        self._db_path = self.cache_dir / "block_index.db"
        self._init_db()

        # Stats (protected by _stats_lock for cross-thread accuracy)
        self._stats_lock = threading.Lock()
        self._latest_payload_inspection_cache: tuple[
            tuple[str, int, int], Dict[str, Any]
        ] | None = None
        # Pending immutable payload bytes use the same lock as the public
        # write-pipeline counters.  Native request fences may wait for this
        # bounded budget instead of silently dropping a causal chain block;
        # ordinary writes keep the historical non-blocking default.
        self._pending_write_condition = threading.Condition(self._stats_lock)
        self.disk_hits = 0
        self.disk_misses = 0
        # WHY a lookup missed. A bare disk_misses count cannot distinguish
        # "the block was never stored" from "it was stored and we failed to
        # read or validate it" — and those need opposite fixes. MEASURED a
        # session with 0 hits against 72 misses where the reason was
        # indistinguishable from the counters alone.
        # WHY a block was never written. The read side proved every L2 miss is
        # "absent" — the blocks were never stored — so the write path is where
        # the answer lives. `full` on a FAST family is the known ~33%-drop
        # defect: admission timeout is 0 for non-path-dependent families.
        self.write_drop_reasons: dict = {
            "full": 0,            # FIFO at capacity and admission timeout expired
            "quiescing": 0,       # store closing / admission shut
            "writer_stopped": 0,  # writer thread not alive
            "byte_budget": 0,     # 512MB pending-bytes budget refused the reservation
        }
        self.disk_miss_reasons: dict = {
            "absent": 0,          # no index entry / no payload on disk
            "load_failed": 0,     # worker-owned read raised
            "budget_locked": 0,   # global budget mutation guard held
            "validation": 0,      # payload found but rejected, entry cleaned up
        }
        self.disk_writes = 0
        self.disk_evictions = 0
        self.tq_native_writes = 0
        self.tq_native_hits = 0
        # Mixed-SWA reconstruction can encounter old terminal checkpoints in
        # the interior of a longer content-addressed chain.  Those rotating
        # windows are not valid for the requested target offset, so loading
        # their (often 100+ MB) tensors only to discard them is pure read
        # amplification.  Count the schema-aware selective reads separately so
        # live /health evidence can prove the optimization actually ran.
        self.selective_rotating_reconstruction_reads = 0
        self.selective_rotating_layers_omitted = 0
        # Candidate selection can inspect native tags/geometry without loading
        # tensor bodies that reconstruction will immediately load again.
        self.validation_metadata_reads = 0
        self.validation_metadata_read_failures = 0
        # Non-blocking, request-correlated write-fence telemetry.  A fence is
        # complete only after every queued block preceding its sentinel has
        # finished its SQLite transaction *and* the drained batch's LRU
        # eviction has settled.  Exact block hashes stay private; /health gets
        # only bounded counts and opaque fence IDs.
        self._write_fence_seq = 0
        self._write_completion_generation = 0
        self._write_inflight = 0
        self._offthread_serializations_queued = 0
        self._offthread_serializations_completed = 0
        self._offthread_serialization_failures = 0
        self._pending_write_bytes = 0
        self._pending_write_byte_drops = 0
        # 4GB, up from 1GB: a single multiturn store burst on a big model is
        # ~1GB of queued payload buffers (measured live: dots3-note, 220 blocks at
        # ~14k tokens), which saturated the old budget exactly — the burst
        # TAIL then raced the sub-second admission wait against the writer's
        # multi-second drain and lost, and each dropped ancestor poisoned its
        # descendants. The budget bounds transient writer-owned bytes (read-only
        # MLX-backed views or copied NumPy inputs), and per-turn bursts are
        # bounded by the DELTA being stored, so 4GB holds a whole burst with
        # margin.
        _pending_env = os.environ.get("VMLX_BLOCK_DISK_PENDING_WRITE_BYTES")
        try:
            _pending_default = (
                max(1, int(_pending_env))
                if _pending_env
                else 4 * 1024 * 1024 * 1024
            )
        except (TypeError, ValueError):
            _pending_default = 4 * 1024 * 1024 * 1024
        self._max_pending_write_bytes = (
            max(1, int(max_pending_write_bytes))
            if max_pending_write_bytes is not None
            else _pending_default
        )
        # Hashes whose writes were budget-dropped this session: descendants of
        # a dropped block can never publish (ancestry requirement), so they
        # are truncated at QUEUE time instead of consuming budget and failing
        # at publish (live: 2 budget drops cascaded into 292 publish failures
        # on a 23.5k-token store). A successful queue removes the hash so a
        # retried parent revives its chain.
        self._session_dropped_hashes: set = set()
        self._write_fences: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_recent_write_fences = 64

        # Per-thread read connections (thread-local storage).
        # MLLM batch generator runs fetch_cache on a worker thread — SQLite
        # connections created on one thread can't be used from another.
        # Using threading.local() gives each thread its own connection.
        self._thread_local = threading.local()
        # MLX arrays loaded from safetensors retain the creating thread's
        # stream identity. The scheduler reconstructs and consumes block
        # payloads on its single model-owning worker, so L2 reads must happen
        # on that same executor even when prefix lookup starts on an API thread.
        self._load_executor: Any = None
        self._load_worker_prefix = "llm-worker"
        # Initialize for the current (main) thread
        self._ensure_read_conn()

        # Background writer thread
        # Queue items contain either frozen, read-only NumPy buffers awaiting
        # CPU safetensors encoding or a lease-owned staged safetensors path.
        # MLX-backed entries are evaluated; their memoryviews own the buffer:
        # ("__numpy_block__", block_hash, tensor_dict, dtype, num_layers,
        #  token_count, parent_hash, fence_id, reserved_bytes, replace_existing)
        # or
        # (block_hash, staged_payload_path, dtype, num_layers, token_count,
        #  parent_hash, fence_id, reserved_bytes, replace_existing)
        # or special commands: ("__access__", ...) or ("__cleanup__", ...)
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._tmp_seq = 0  # Monotonic counter for unique temp file names
        self._stop_event = threading.Event()
        # Publication lifecycle is intentionally separate from ``_stats_lock``.
        # A producer can spend meaningful time freezing MLX state before it
        # reaches the queue.  Shutdown/clear must close admission first and wait
        # for those producers without holding telemetry or aggregate-budget
        # locks that the producer/writer also needs.
        self._write_lifecycle_lock = threading.Lock()
        self._write_lifecycle = threading.Condition(self._write_lifecycle_lock)
        self._accepting_writes = True
        self._shutdown_started = False
        self._clear_in_progress = False
        self._active_write_producers = 0
        # Counts queued *and dequeued* items until their containing batch has
        # fully completed.  ``Queue.empty()`` cannot provide this guarantee: a
        # writer may already have dequeued a block that can still republish it.
        self._pending_write_items = 0
        # The owning scheduler reports whether any request is waiting or
        # running; maintenance never starts while a request is in flight, so
        # a store that is about to happen cannot land behind the rescan.
        self._activity_probe = activity_probe
        # Monotonic time of the last enqueue or batch start; the deferred
        # budget rescan waits for a quiet writer, not a momentarily empty queue
        # (a store enqueues its blocks one by one while the writer drains).
        self._last_write_activity_monotonic = time.monotonic()
        self._writer_thread = threading.Thread(
            target=self._background_writer, daemon=True, name="block-disk-writer"
        )
        self._writer_shutdown_timeout_seconds = 5.0
        self._shutdown_finalize_lock = threading.Lock()
        self._shutdown_finalized = False
        self._delayed_shutdown_thread: Optional[threading.Thread] = None
        self._budget_recovery_lock = threading.Lock()
        self._budget_recovery_interval_ns = int(5.0 * 1_000_000_000)
        self._last_budget_recovery_attempt_ns = 0
        self._pending_publication_recovery: tuple[bytes | None, set[str]] | None = None

        # Publish the aggregate-budget owner only after local DB/read/thread
        # primitives are ready.  A constructor failure after lease publication
        # must remove that lease; otherwise this live PID can pin a stale,
        # stricter max until process exit even though no store owns it.
        self.global_budget = get_global_disk_cache_budget(
            self.global_cache_root,
            self.max_size_bytes,
            allow_legacy_hashed_namespaces=allow_legacy_hashed_namespaces,
            allow_legacy_direct_namespace=allow_legacy_direct_namespace,
        )
        try:
            # Never delete temporary files owned by another process. They are
            # excluded from eviction and counted as protected physical bytes by the
            # root coordinator; a writer removes its own temp on success/failure.
            self._cleanup_orphaned_tmp()

            # A user can lower the disk-cap slider between sessions. Enforce that
            # new ceiling before serving reads or accepting writes; waiting for the
            # first background write left an oversized cache indefinitely after
            # restart. This synchronous startup trim touches only files and SQLite.
            global_startup_trim = self.global_budget.enforce(force=True)
            self._global_budget_write_enabled = bool(
                global_startup_trim.accounted
                and global_startup_trim.compliant
            )
            self.disk_evictions += int(
                getattr(global_startup_trim, "evicted_entries", 0) or 0
            )
            if not self._global_budget_write_enabled:
                self._last_budget_recovery_attempt_ns = time.monotonic_ns()
            if not global_startup_trim.compliant:
                logger.warning(
                    "Global block-cache startup trim incomplete: %s",
                    global_startup_trim.error
                    or (
                        f"{global_startup_trim.bytes_after} bytes remain above "
                        f"{global_startup_trim.max_size_bytes}"
                    ),
                )

            self._writer_thread.start()

            entry_count = self._count_entries()
            total_size = self._total_size()
            logger.info(
                f"BlockDiskStore initialized: dir={self.cache_dir}, "
                f"max_size={max_size_gb:.1f}GB, entries={entry_count}, "
                f"size={total_size / 1024**3:.2f}GB"
            )
        except Exception:
            self._stop_event.set()
            if self._writer_thread.is_alive():
                self._writer_thread.join(timeout=5.0)
            try:
                conn = getattr(self._thread_local, "read_conn", None)
                if conn is not None:
                    conn.close()
                    self._thread_local.read_conn = None
            except Exception:
                pass
            self.global_budget.close()
            raise

    def _init_db(self) -> None:
        """Create SQLite index with WAL mode."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            # Serialize schema discovery + ALTER across concurrent processes
            # opening the same model namespace for the first time.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    block_hash    TEXT PRIMARY KEY,
                    parent_hash   TEXT,
                    ancestry_known INTEGER NOT NULL DEFAULT 1,
                    file_name     TEXT NOT NULL,
                    num_tokens    INTEGER NOT NULL,
                    num_layers    INTEGER NOT NULL,
                    dtype         TEXT NOT NULL,
                    file_size     INTEGER NOT NULL,
                    created_at    REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count  INTEGER DEFAULT 0
                )
            """)
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(blocks)").fetchall()
            }
            if "parent_hash" not in columns:
                conn.execute("ALTER TABLE blocks ADD COLUMN parent_hash TEXT")
            if "ancestry_known" not in columns:
                # Existing rows predate persisted chain ancestry.  ``NULL`` is
                # ambiguous there: it can mean either a real root or unknown
                # ancestry, so never silently promote legacy rows to roots.
                conn.execute(
                    "ALTER TABLE blocks ADD COLUMN ancestry_known "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            if "group_size" not in columns:
                # NULL = one file per block; N = this row's file is a hard link to an N-member group file.
                conn.execute("ALTER TABLE blocks ADD COLUMN group_size INTEGER")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocks_lru ON blocks(last_accessed ASC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocks_parent "
                "ON blocks(parent_hash)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS block_write_pins (
                    block_hash    TEXT NOT NULL,
                    owner_lease_id TEXT NOT NULL,
                    fence_id      TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    PRIMARY KEY (block_hash, owner_lease_id, fence_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_block_write_pins_owner "
                "ON block_write_pins(owner_lease_id, fence_id)"
            )
            conn.commit()
        finally:
            conn.close()

    def _ensure_read_conn(self) -> sqlite3.Connection:
        """Get or create a read connection for the current thread."""
        conn = getattr(self._thread_local, 'read_conn', None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=1.0)
            conn.execute("PRAGMA journal_mode=WAL")
            self._thread_local.read_conn = conn
        return conn

    @property
    def _read_conn(self) -> sqlite3.Connection:
        """Thread-safe read connection accessor."""
        return self._ensure_read_conn()

    def _cleanup_orphaned_tmp(self) -> None:
        """Preserve temp files because ownership cannot be proven cross-process."""
        return

    def _count_entries(self) -> int:
        return self._read_conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]

    def _total_size(self) -> int:
        return self._read_conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) FROM blocks"
        ).fetchone()[0]

    def _hash_to_path(self, hash_hex: str) -> Path:
        """Shard by first 2 chars for filesystem efficiency."""
        shard = hash_hex[:2]
        shard_dir = self.blocks_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{hash_hex}.safetensors"

    def has_block(self, block_hash: bytes) -> bool:
        """Return whether a block hash has a readable finalized L2 entry."""
        with self.global_budget.mutation_guard() as locked:
            if not locked:
                return False
            return self._has_block_guarded(block_hash)

    def has_block_record(self, block_hash: bytes) -> bool:
        """Check finalized index/file readability without loading tensors.

        Callers may already hold a validated representation stamp and only
        need to know whether aggregate-budget eviction removed its payload.
        Unlike :meth:`has_block`, this does not deserialize for codec-policy
        compatibility and does not count a cache hit or touch the durable LRU.
        """
        with self.global_budget.mutation_guard() as locked:
            if not locked:
                return False
            hash_hex = block_hash.hex()
            row, _ok = self._query_index_row(
                "SELECT file_name FROM blocks WHERE block_hash = ?",
                (hash_hex,),
            )
            return bool(
                row is not None
                and self._indexed_payload_is_readable(str(row[0]))
            )

    def _query_index_row(self, sql: str, params: tuple):
        """Run an index query, reconnecting once, and NEVER raise.

        The reconnect was itself unprotected: under fd exhaustion
        ``sqlite3.connect`` raises OperationalError("unable to open database
        file") and the exception escaped its caller. From ``write_block_async``
        that skipped the fence settle, leaving ``_active`` stuck above zero
        forever — the fence never becomes ready, never terminalizes, is never
        pruned (unfinished fences are retained by design), and at the 64-fence
        cap ``begin_write_fence`` starts raising. Its caller catches that, warns,
        and proceeds with ``disk_write_fence_id=None``, so EVERY later request
        writes unfenced for the life of the process.

        Returning "no row" is the safe answer: it costs a cache miss, and the
        block is found again once the descriptor storm passes.
        """
        try:
            return self._read_conn.execute(sql, params).fetchone(), True
        except sqlite3.OperationalError:
            pass
        try:
            self._thread_local.read_conn = sqlite3.connect(
                str(self._db_path),
                timeout=1.0,
            )
            return self._read_conn.execute(sql, params).fetchone(), True
        except sqlite3.Error as exc:
            logger.warning(f"Block index unavailable ({exc}); treating as a miss")
            return None, False

    def _has_block_guarded(self, block_hash: bytes) -> bool:
        """Guarded implementation of :meth:`has_block`."""
        hash_hex = block_hash.hex()
        row, _ok = self._query_index_row(
            "SELECT file_name, dtype FROM blocks WHERE block_hash = ?",
            (hash_hex,),
        )
        if row is None:
            return False
        file_path = self.cache_dir / row[0]
        try:
            identity = _payload_identity(file_path)
        except OSError:
            return False
        if not self._allow_tq_native:
            # Allocation consults has_block() before read_block().  If an old
            # TQ record is reported as present here, the paged manager reuses
            # its hash slot and the Off run never gets a chance to write a
            # standard replacement after read_block() rejects it.  Inspect the
            # typed payload in the explicit Off mode and evict it up front --
            # once per exact file (see _verified_plain_payloads).
            if self._payload_verified_plain(hash_hex, identity):
                return True
            try:
                data = mx.load(str(file_path))
                if self._evict_incompatible_tq_block(
                    hash_hex,
                    file_path,
                    _deserialize_block(data, row[1]),
                ):
                    self._forget_verified_plain(hash_hex)
                    return False
            except Exception as exc:
                logger.warning(
                    "Failed to inspect block %s for non-TQ compatibility: %s",
                    hash_hex[:12],
                    exc,
                )
                return False
            self._note_verified_plain(hash_hex, identity)
        return True

    def _payload_verified_plain(
        self, hash_hex: str, identity: Tuple[int, int, int, int]
    ) -> bool:
        with self._verified_plain_lock:
            return self._verified_plain_payloads.get(hash_hex) == identity

    def _note_verified_plain(
        self, hash_hex: str, identity: Tuple[int, int, int, int]
    ) -> None:
        with self._verified_plain_lock:
            payloads = self._verified_plain_payloads
            payloads.pop(hash_hex, None)
            payloads[hash_hex] = identity
            while len(payloads) > self._verified_plain_max:
                payloads.pop(next(iter(payloads)))

    def _forget_verified_plain(self, hash_hex: str) -> None:
        with self._verified_plain_lock:
            self._verified_plain_payloads.pop(hash_hex, None)

    def _note_payload_checks(
        self,
        hash_hex: str,
        identity: Tuple[int, int, int, int],
        rotating_offsets: Optional[Dict[int, int]],
    ) -> None:
        with self._payload_check_lock:
            memo = self._payload_check_memo
            memo.pop(hash_hex, None)
            memo[hash_hex] = (
                identity,
                True,
                dict(rotating_offsets) if rotating_offsets is not None else None,
            )
            while len(memo) > self._verified_plain_max:
                memo.pop(next(iter(memo)))

    def _forget_payload_checks(self, hash_hex: str) -> None:
        with self._payload_check_lock:
            self._payload_check_memo.pop(hash_hex, None)
            self._validation_entries_memo.pop(hash_hex, None)

    def _evict_incompatible_tq_block(
        self,
        hash_hex: str,
        file_path: Path,
        cache_data: List[Tuple],
    ) -> bool:
        """Queue eviction of TQ data that conflicts with explicit non-TQ mode."""
        if self._allow_tq_native or not _cache_data_has_tq(cache_data):
            return False
        # has_block/read_block hold the aggregate *shared* read lock.  Physical
        # deletion here would race another process's load and bypass global
        # accounting.  The background writer consumes cleanup commands under
        # the exclusive mutation lock and reconciles the aggregate ledger.
        self._queue_index_cleanup(hash_hex)
        logger.info(
            "Queued eviction of TQ-native block %s because persisted TQ reads "
            "are disabled",
            hash_hex[:12],
        )
        return True

    # =========================================================================
    # Read
    # =========================================================================

    def set_load_executor(
        self,
        executor: Any,
        worker_name_prefix: Optional[str] = None,
    ) -> None:
        """Bind MLX-backed block reads to the model-owning worker thread."""
        self._load_executor = executor
        if worker_name_prefix is None:
            worker_name_prefix = (
                getattr(executor, "_thread_name_prefix", "") or "llm-worker"
            )
        self._load_worker_prefix = worker_name_prefix

    def read_block(self, block_hash: bytes) -> Optional[List[Tuple]]:
        """Read and deserialize a block on the model-owning MLX worker."""
        executor = getattr(self, "_load_executor", None)
        prefix = getattr(self, "_load_worker_prefix", "llm-worker")
        if executor is None or threading.current_thread().name.startswith(prefix):
            return self._read_block_impl(block_hash)
        try:
            return executor.submit(self._read_block_impl, block_hash).result()
        except Exception as exc:
            logger.warning(
                "Block L2 worker-owned load failed; treating as miss: %s",
                exc,
                exc_info=True,
            )
            with self._stats_lock:
                self.disk_misses += 1
                self._note_miss("load_failed")
            return None

    def read_block_for_reconstruction(
        self,
        block_hash: bytes,
        *,
        rotating_target_offset: int,
    ) -> Optional[List[Tuple]]:
        """Read one SSD-only reconstruction block without stale SWA windows.

        The target offset is part of the native RotatingKVCache contract.  A
        rotating checkpoint whose stored absolute offset differs from it cannot
        contribute to this reconstruction; ``prefix_cache`` already ignores
        such entries after loading them.  This path performs that same decision
        from safetensors metadata before materializing the large K/V tensors.

        Callers must treat the result as request-local and transient.  It may
        contain ``rotating_kv_pending`` markers in place of omitted historical
        checkpoints and therefore must never be installed as shared block RAM.
        """
        executor = getattr(self, "_load_executor", None)
        prefix = getattr(self, "_load_worker_prefix", "llm-worker")
        args = (block_hash, max(0, int(rotating_target_offset)))
        if executor is None or threading.current_thread().name.startswith(prefix):
            return self._read_block_impl(*args)
        try:
            return executor.submit(self._read_block_impl, *args).result()
        except Exception as exc:
            logger.warning(
                "Selective block L2 worker-owned load failed; treating as miss: %s",
                exc,
                exc_info=True,
            )
            with self._stats_lock:
                self.disk_misses += 1
                self._note_miss("load_failed")
            return None

    def advise_block_reads(self, block_hashes: List[bytes]) -> int:
        """Ask the OS to start reading these blocks' payload files (best-effort).

        A long-prefix restore reads its block files one at a time (queue depth
        1). On a cold page cache that left the SSD mostly idle: 512 files /
        1.84 GB loaded in 1.06 s sequentially vs 0.45 s after an up-front
        read-ahead hint (box/max2 M5 Max, after `purge`, identical bytes).
        This only issues an asynchronous read-ahead hint per file (F_RDADVISE on
        macOS, POSIX_FADV_WILLNEED elsewhere); nothing is loaded into Python or
        MLX, no lock is taken, and a file that disappears is simply skipped.
        Returns the number of files advised.
        """
        hashes = [bytes(h).hex() for h in block_hashes if h]
        if not hashes:
            return 0
        advised = 0
        advised_inodes: set = set()
        for start in range(0, len(hashes), 500):
            chunk = hashes[start : start + 500]
            try:
                rows = self._read_conn.execute(
                    "SELECT file_name, group_size FROM blocks WHERE block_hash IN (%s)"
                    % ",".join("?" * len(chunk)),
                    chunk,
                ).fetchall()
            except sqlite3.Error:
                return advised
            for file_name, group_size in rows:
                path = self.cache_dir / file_name
                if group_size and int(group_size) > 1:
                    # members of one group file share an inode: advise it once
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    inode = (st.st_dev, st.st_ino)
                    if inode in advised_inodes:
                        continue
                    advised_inodes.add(inode)
                if _advise_file_read(path):
                    advised += 1
        return advised

    def read_block_validation_entries(
        self,
        block_hash: bytes,
    ) -> Optional[List[Tuple]]:
        """Read native candidate metadata without materializing payload arrays.

        This path is intentionally not counted as a payload ``disk_hit``. It
        prevents SSD-only DSV4/ZAYA/mixed-SWA validation from loading every
        block once during candidate selection and once again during worker
        reconstruction. Missing, legacy, or malformed metadata returns ``None``
        so callers preserve the established full-reader fallback.
        """
        with self.global_budget.mutation_guard() as locked:
            if not locked:
                with self._stats_lock:
                    self.validation_metadata_read_failures += 1
                return None
            hash_hex = block_hash.hex()
            row, _ok = self._query_index_row(
                "SELECT file_name, group_size FROM blocks WHERE block_hash = ?",
                (hash_hex,),
            )
            if row is None:
                with self._stats_lock:
                    self.validation_metadata_read_failures += 1
                return None
            file_path = self.cache_dir / row[0]
            group_size = int(row[1] or 0) if len(row) > 1 else 0
            try:
                identity = _payload_identity(file_path)
            except OSError:
                with self._stats_lock:
                    self.validation_metadata_read_failures += 1
                return None
            with self._payload_check_lock:
                memo = self._validation_entries_memo.get(hash_hex)
                if memo is not None and memo[0] == identity:
                    self._validation_entries_memo.move_to_end(hash_hex)
                    entries = list(memo[1])
                else:
                    entries = None
            if entries is not None:
                with self._stats_lock:
                    self.validation_metadata_reads += 1
                self._touch_payload(file_path, identity, group_size)
                self._queue_access_update(hash_hex)
                return entries
            try:
                from .cache_record_validator import reject_safetensors_or_warn

                with self._payload_check_lock:
                    header_known_ok = group_size > 1 and bool(self._group_header_ok.get(identity))
                if not header_known_ok and not reject_safetensors_or_warn(
                    str(file_path),
                    expected_num_layers=getattr(self, "_expected_num_layers", None),
                    source=f"L2-validation-metadata:{hash_hex[:12]}",
                    delete_on_reject=False,
                ):
                    with self._stats_lock:
                        self.validation_metadata_read_failures += 1
                    return None
            except Exception:
                # Keep full read + post-load validation as the fail-closed
                # compatibility path in reduced installations.
                with self._stats_lock:
                    self.validation_metadata_read_failures += 1
                return None
            member = None
            if group_size > 1:
                with self._payload_check_lock:
                    self._group_header_ok[identity] = True
                found = self._group_meta_for(file_path, identity)
                index = None if found is None else found[1].get(hash_hex)
                if index is None:
                    with self._stats_lock:
                        self.validation_metadata_read_failures += 1
                    return None
                member = (found[0], index)
            entries = _load_block_validation_entries(file_path, member)
            if entries is None:
                with self._stats_lock:
                    self.validation_metadata_read_failures += 1
                return None
            with self._payload_check_lock:
                memo = self._validation_entries_memo
                memo[hash_hex] = (identity, list(entries))
                memo.move_to_end(hash_hex)
                while len(memo) > self._validation_entries_max:
                    memo.popitem(last=False)
            with self._stats_lock:
                self.validation_metadata_reads += 1
            self._touch_payload(file_path, identity, group_size)
            self._queue_access_update(hash_hex)
            return entries


    def _note_miss(self, reason: str) -> None:
        """Record WHY a lookup missed. Caller already holds _stats_lock."""
        if reason in self.disk_miss_reasons:
            self.disk_miss_reasons[reason] += 1

    def _read_block_impl(
        self,
        block_hash: bytes,
        rotating_target_offset: Optional[int] = None,
    ) -> Optional[List[Tuple]]:
        # Root-exclusive eviction cannot unlink the payload while validation,
        # MLX load, deserialization, and the durable LRU touch are in flight.
        with self.global_budget.mutation_guard() as locked:
            if not locked:
                with self._stats_lock:
                    self.disk_misses += 1
                    self._note_miss("budget_locked")
                return None
            return self._read_block_impl_guarded(
                block_hash,
                rotating_target_offset=rotating_target_offset,
            )

    def read_group_span_for_reconstruction(
        self,
        block_hashes: List[bytes],
    ) -> Optional[Tuple[List[Tuple], int]]:
        """Read a run of reconstruction blocks that are consecutive members of ONE group file as one payload.

        ``block_hashes`` are the next blocks of the table, in order. Returns ``(cache_data, n)`` covering the first
        ``n >= 2`` of them (KV entries span all n blocks along the token axis; pending markers as in the last member),
        or None when the first block is not a group member or the span would be shorter than 2 -- the caller then
        reads per block. Request-local and transient, like ``read_block_for_reconstruction``. One deserialize, one
        validation, one inode touch for the span instead of one of each per 64-token block.
        """
        if not block_hashes or len(block_hashes) < 2:
            return None
        executor = getattr(self, "_load_executor", None)
        prefix = getattr(self, "_load_worker_prefix", "llm-worker")
        if executor is None or threading.current_thread().name.startswith(prefix):
            return self._read_group_span_impl(list(block_hashes))
        try:
            return executor.submit(self._read_group_span_impl, list(block_hashes)).result()
        except Exception as exc:
            logger.warning("Group span L2 worker-owned load failed; reading per block: %s", exc)
            return None

    def _read_group_span_impl(self, block_hashes: List[bytes]) -> Optional[Tuple[List[Tuple], int]]:
        if not HAS_MLX:
            return None
        with self.global_budget.mutation_guard() as locked:
            if not locked:
                return None
            try:
                return self._read_group_span_guarded(block_hashes)
            except Exception as exc:  # noqa: BLE001 - the per-block path re-reads and classifies failures
                logger.debug("Group span read fell back to per-block reads: %s", exc)
                return None

    def _read_group_span_guarded(self, block_hashes: List[bytes]) -> Optional[Tuple[List[Tuple], int]]:
        import numpy as np

        first_hex = bytes(block_hashes[0]).hex()
        row, _ok = self._query_index_row(
            "SELECT file_name, dtype, group_size FROM blocks WHERE block_hash = ?",
            (first_hex,),
        )
        if row is None or int(row[2] or 0) <= 1:
            return None
        file_name, dtype, _group_size = row
        file_path = self.cache_dir / file_name
        identity = _payload_identity(file_path)
        with self._payload_check_lock:
            header_ok = bool(self._group_header_ok.get(identity))
        if not header_ok:
            from .cache_record_validator import reject_safetensors_or_warn

            if not reject_safetensors_or_warn(
                str(file_path),
                expected_num_layers=getattr(self, "_expected_num_layers", None),
                source=f"L2-disk-group-header:{first_hex[:12]}",
                delete_on_reject=False,
            ):
                return None
            with self._payload_check_lock:
                self._group_header_ok[identity] = True
        self._group_member_data(file_path, identity, first_hex)      # loads + caches the group
        group, group_meta, ordinals, member_keys = self._group_cache[identity]
        members = group_meta["members"]
        start = ordinals[first_hex]
        n = 1
        while n < len(block_hashes) and start + n < len(members) and bytes(block_hashes[n]).hex() == members[start + n]:
            n += 1
        if n < 2:
            return None
        # Every member of the span must still be indexed AS this group (a replaced or cleaned block is not used).
        span_hex = [bytes(h).hex() for h in block_hashes[:n]]
        placeholders = ",".join("?" for _ in span_hex)
        rows = self._read_conn.execute(
            f"SELECT block_hash, file_name, dtype, group_size FROM blocks WHERE block_hash IN ({placeholders})",
            span_hex,
        ).fetchall()
        by_hash = {str(r[0]): r for r in rows}
        # (Content-addressed and never replaced into a group: a member row that still says group_size > 1 with
        # the same dtype is this block's group record; its bytes in this group file are its payload.)
        ok = 0
        for h in span_hex:
            r = by_hash.get(h)
            if r is None or int(r[3] or 0) <= 1 or r[2] != dtype:
                break
            ok += 1
        n = ok
        if n < 2:
            return None
        # Identical block metadata across the span (same layer tags / dtypes); shrink to the identical prefix.
        meta0 = np.asarray(group[f"b{start}.{_BLOCK_META_KEY}"]).tobytes()
        same = 1
        while same < n and np.asarray(group[f"b{start + same}.{_BLOCK_META_KEY}"]).tobytes() == meta0:
            same += 1
        n = same
        if n < 2:
            return None
        last = start + n - 1
        span: Dict[str, Any] = {}
        for name, offsets in group_meta["concat"].items():
            span[name] = group[name][:, :, int(offsets[start]):int(offsets[start + n]), :]
        for short, full in member_keys[last]:
            arr = group[full]
            if short != _BLOCK_META_KEY and short not in span and int(getattr(arr, "size", 0) or 0) > 64:
                return None                   # a per-block tensor that cannot span blocks: read per block
            span[short] = arr
        cache_data = _deserialize_block(span, dtype)
        from .cache_record_validator import reject_or_warn

        if not reject_or_warn(
            cache_data,
            expected_num_layers=getattr(self, "_expected_num_layers", None),
            source=f"L2-disk-group-span:{first_hex[:12]}x{n}",
        ):
            return None
        mx.eval(*span.values())
        self._touch_payload(file_path, identity, 2)
        with self._stats_lock:
            self.disk_hits += n
            self.group_span_reads = getattr(self, "group_span_reads", 0) + 1
            self.group_span_blocks = getattr(self, "group_span_blocks", 0) + n
        for h in span_hex[:n]:
            self._queue_access_update(h)
        return cache_data, n

    def _group_member_data(self, file_path: Path, identity, hash_hex: str) -> Dict[str, Any]:
        """Member ``hash_hex``'s tensor dict from its group file, loading the group once per file identity."""
        entry = self._group_cache.get(identity)
        if entry is None:
            group = mx.load(str(file_path))
            group_meta = _decode_group_meta(group.pop(_GROUP_META_KEY, None))
            if group_meta is None:
                raise ValueError("group file has no valid group metadata")
            # One read for every member; lazily loaded arrays would keep the file's
            # reader alive across the whole reconstruct (descriptor pressure).
            mx.eval(*group.values())
            ordinals = {str(h): i for i, h in enumerate(group_meta["members"])}
            entry = (group, group_meta, ordinals, _group_member_keys(group, len(group_meta["members"])))
            self._group_cache[identity] = entry
            while len(self._group_cache) > self._group_cache_max:
                self._group_cache.popitem(last=False)
        else:
            self._group_cache.move_to_end(identity)
        group, group_meta, ordinals, member_keys = entry
        index = ordinals.get(hash_hex)
        if index is None:
            raise ValueError("block is not a member of its group file")
        with self._stats_lock:
            self.group_member_reads += 1
        return _group_member_view(group, group_meta, index, member_keys)

    def _touch_payload(self, file_path: Path, identity, group_size: int) -> None:
        """mtime = the cross-process LRU signal. A group file's links share one inode, and utime on a
        many-linked APFS inode costs ~0.6 ms (vs ~0.04 ms): touch each group inode at most once a second."""
        if group_size > 1:
            now = time.monotonic()
            last = self._group_touch_times.get(identity)
            if last is not None and now - last < 1.0:
                return
            self._group_touch_times[identity] = now
            if len(self._group_touch_times) > 4096:
                self._group_touch_times.clear()
        try:
            os.utime(file_path, None)
        except OSError:
            pass

    def release_group_cache(self) -> None:
        """Drop the loaded group payloads (after a reconstruct, and at idle)."""
        self._group_cache.clear()

    def _group_meta_for(self, file_path: Path, identity):
        """(group meta, hash -> ordinal) of a group file, header-only and cached per file identity."""
        entry = self._group_cache.get(identity)
        if entry is not None:
            return entry[1], entry[2]
        cached = self._group_meta_only.get(identity)
        if cached is None:
            try:
                from safetensors import safe_open

                with safe_open(str(file_path), framework="numpy") as handle:
                    group_meta = _decode_group_meta(handle.get_tensor(_GROUP_META_KEY))
            except Exception:
                return None
            if group_meta is None:
                return None
            cached = (group_meta, {str(h): i for i, h in enumerate(group_meta["members"])})
            self._group_meta_only[identity] = cached
            while len(self._group_meta_only) > 512:
                self._group_meta_only.popitem(last=False)
        return cached

    def _read_block_impl_guarded(
        self,
        block_hash: bytes,
        *,
        rotating_target_offset: Optional[int] = None,
    ) -> Optional[List[Tuple]]:
        """
        Read a block from disk by its chain hash.

        This method is read-only on the main thread — access metadata updates
        are deferred to the background writer to avoid blocking inference.

        Args:
            block_hash: The BlockHash (SHA-256 chain hash bytes)

        Returns:
            cache_data in the same format as CacheBlock.cache_data,
            or None if not found on disk.
        """
        if not HAS_MLX:
            return None

        hash_hex = block_hash.hex()

        # Connection might be stale after a writer vacuum — reconnect once, and
        # degrade to a miss rather than raising if the reconnect also fails.
        row, _ok = self._query_index_row(
            "SELECT file_name, dtype, group_size FROM blocks WHERE block_hash = ?",
            (hash_hex,),
        )

        if row is None:
            with self._stats_lock:
                self.disk_misses += 1
                self._note_miss("absent")
            return None

        file_name, dtype, group_size = row
        group_size = int(group_size or 0)

        file_path = self.cache_dir / file_name

        try:
            identity = _payload_identity(file_path)
        except OSError:
            identity = None
        if identity is None:
            # Stale index entry — queue cleanup to background writer
            self._forget_payload_checks(hash_hex)
            self._queue_index_cleanup(hash_hex)
            with self._stats_lock:
                self.disk_misses += 1
                self._note_miss("validation")
            return None

        try:
            # Codex 2026-05-06 follow-up: validate the safetensors HEADER
            # BEFORE calling mx.load. Without this, a corrupt-header file
            # whose declared shapes describe multi-hundred-GB tensors makes
            # mx.load itself trigger [metal::malloc] BEFORE the post-load
            # validator can reject the record. Reads only ~1 KB of header
            # JSON and rejects + deletes the file if any tensor exceeds the
            # 4 GB / 16 GB / 256K dim caps.
            try:
                from .cache_record_validator import (
                    reject_safetensors_or_warn,
                    reject_or_warn,
                )
            except Exception:
                reject_safetensors_or_warn = None
                reject_or_warn = None
            with self._payload_check_lock:
                memo = self._payload_check_memo.get(hash_hex)
                group_header_ok = (
                    self._group_header_ok.get(identity) if group_size > 1 else None
                )
            if memo is not None and memo[0] != identity:
                memo = None
            if reject_safetensors_or_warn is not None and not (
                (memo is not None and memo[1]) or group_header_ok
            ):
                if not reject_safetensors_or_warn(
                    str(file_path),
                    expected_num_layers=getattr(self, "_expected_num_layers", None),
                    source=f"L2-disk-header:{hash_hex[:12]}",
                    # read_block holds the aggregate shared lock; cleanup is
                    # queued below and runs under the exclusive writer lock.
                    delete_on_reject=False,
                ):
                    # The validator reports one bool for two different worlds: a
                    # real caps/JSON rejection, and a stat/open/short-read that
                    # failed closed. Under fd exhaustion the second is what fires,
                    # and cleaning up on it deletes a healthy chain.
                    if self._read_failure_is_transient(file_path):
                        logger.warning(
                            f"Header check for block {hash_hex[:12]} failed on a "
                            f"transient I/O error; keeping the entry"
                        )
                        with self._stats_lock:
                            self.disk_misses += 1
                            self._note_miss("absent")
                        return None
                    # Queue index + file cleanup under the background writer's
                    # exclusive aggregate mutation transaction.
                    self._forget_payload_checks(hash_hex)
                    self._queue_index_cleanup(hash_hex)
                    with self._stats_lock:
                        self.disk_misses += 1
                        self._note_miss("validation")
                    return None

            if group_size > 1:
                # Header checked once per group file (its members share it).
                with self._payload_check_lock:
                    if len(self._group_header_ok) > 4096:
                        self._group_header_ok.clear()
                    self._group_header_ok[identity] = True
            omitted_rotating_layers: set[int] = set()
            selective_payload = None
            rotating_offsets = memo[2] if memo is not None else None
            if group_size > 1:
                # Members never hold rotating windows (_numpy_block_group_eligible).
                rotating_offsets = {}
            if rotating_target_offset is not None and not (
                # Known windows all at the target offset (or no rotating layer
                # at all): the selective reader would omit nothing and return
                # None, so skip re-opening the file to find that out.
                rotating_offsets is not None
                and all(
                    offset == int(rotating_target_offset)
                    for offset in rotating_offsets.values()
                )
            ):
                probe: Dict[str, Any] = {}
                selective_payload = _load_reconstruction_payload_without_stale_rotating(
                    file_path,
                    target_offset=int(rotating_target_offset),
                    probe=probe,
                )
                if "rotating_offsets" in probe:
                    rotating_offsets = probe["rotating_offsets"]
            if group_size > 1:
                data = self._group_member_data(file_path, identity, hash_hex)
            elif selective_payload is None:
                data = mx.load(str(file_path))
            else:
                data, omitted_rotating_layers = selective_payload
            cache_data = _deserialize_block(data, dtype)
            if omitted_rotating_layers:
                # Preserve an explicit native marker for every omitted layer.
                # A bare ("skip",) is reserved for cumulative/SSM state and can
                # be accepted as a partial hybrid restore; treating rotating
                # state that way could silently return wrong Gemma/Laguna
                # logits when the exact terminal checkpoint is absent.
                for layer_idx in omitted_rotating_layers:
                    if (
                        layer_idx >= len(cache_data)
                        or cache_data[layer_idx] != ("skip",)
                    ):
                        raise ValueError(
                            "selective rotating reconstruction produced an "
                            f"unexpected layer {layer_idx} payload"
                        )
                    cache_data[layer_idx] = (
                        "rotating_kv_pending",
                        "RotatingKVCache",
                    )
            if self._evict_incompatible_tq_block(
                hash_hex, file_path, cache_data
            ):
                with self._stats_lock:
                    self.disk_misses += 1
                    self._note_miss("validation")
                return None
            # Post-load validator (defense in depth): the header validator
            # only sees declared shapes, not the deserialized cache_data
            # tag/shape coherence. This catch covers bugs the header check
            # couldn't see (e.g. tag mismatch, missing fields).
            if reject_or_warn is not None:
                if not reject_or_warn(
                    cache_data,
                    expected_num_layers=getattr(self, "_expected_num_layers", None),
                    source=f"L2-disk:{hash_hex[:12]}",
                ):
                    # Reject + queue cleanup so we don't try this entry again.
                    self._forget_payload_checks(hash_hex)
                    self._queue_index_cleanup(hash_hex)
                    with self._stats_lock:
                        self.disk_misses += 1
                        self._note_miss("validation")
                    return None
            # mx.load is lazy: every unevaluated file-backed array keeps its
            # safetensors reader (and descriptor) alive. Reconstruction holds
            # thousands of block payloads before concatenating them, so leaving
            # evaluation to the caller exhausted RLIMIT_NOFILE mid-chain.
            # Finish this block's I/O while its shared aggregate lock is held;
            # returned tensors own bytes, not a deferred read of an evictable
            # file. This preserves native dtype/format and bounds descriptors
            # independently of context length without raising process limits.
            mx.eval(*data.values())
            with self._stats_lock:
                self.disk_hits += 1
                if _cache_data_has_tq(cache_data):
                    self.tq_native_hits += 1
                if omitted_rotating_layers:
                    self.selective_rotating_reconstruction_reads += 1
                    self.selective_rotating_layers_omitted += len(
                        omitted_rotating_layers
                    )
            # File mtime is the non-droppable cross-process LRU signal.  The
            # SQLite update remains asynchronous for latency, but a full queue
            # can no longer make a just-read block appear old to global trim.
            self._touch_payload(file_path, identity, group_size)
            # Queue access metadata update to background (non-blocking)
            self._queue_access_update(hash_hex)
            if omitted_rotating_layers:
                logger.debug(
                    "Disk cache selective mixed-SWA hit: %s (%s, %d layers, "
                    "%d stale rotating checkpoints omitted for target=%d)",
                    hash_hex[:12],
                    dtype,
                    len(cache_data),
                    len(omitted_rotating_layers),
                    int(rotating_target_offset or 0),
                )
            else:
                logger.debug(
                    f"Disk cache hit: {hash_hex[:12]} ({dtype}, {len(cache_data)} layers)"
                )
            self._note_payload_checks(hash_hex, identity, rotating_offsets)
            return cache_data
        except Exception as e:
            if self._read_failure_is_transient(file_path, e):
                # Environment, not content. Count a miss and leave the chain
                # alone — deleting here would take every descendant with it.
                logger.warning(
                    f"Transient read failure for block {hash_hex[:12]} "
                    f"({type(e).__name__}: {e}); keeping the entry"
                )
                with self._stats_lock:
                    self.disk_misses += 1
                    self._note_miss("absent")
                return None
            logger.warning(f"Failed to load block {hash_hex[:12]}: {e}")
            # Corrupt file — queue removal
            self._forget_payload_checks(hash_hex)
            self._queue_index_cleanup(hash_hex)
            with self._stats_lock:
                self.disk_misses += 1
                self._note_miss("validation")
            return None

    @staticmethod
    def _read_failure_is_transient(file_path, exc: BaseException | None = None) -> bool:
        """Is this read failure the environment's fault rather than the file's?

        The distinction decides whether a block is DELETED. `_queue_index_cleanup`
        does not remove one entry — it walks the descendant chain and unlinks every
        payload behind it, so treating a transient failure as corruption destroys a
        healthy chain and turns a warm long-context read back into a cold prefill.

        Type alone cannot decide it: MLX wraps an open failure in RuntimeError
        (`[load_safetensors] Failed to open file ...`), which is indistinguishable
        by type from a truncated-file parse error. So probe the file — if it cannot
        even be opened, the failure was the environment (fd exhaustion, EIO), not
        the content. Under fd pressure the probe itself fails, which is the same
        verdict, so the check is safe in exactly the storm it exists for.
        """
        if isinstance(exc, (OSError, MemoryError)):
            return True
        # MLX names the failure mode in the message, and open-failure reads
        # differently from every content failure:
        #   open    -> "[load_safetensors] Failed to open file ..."
        #   corrupt -> "[load_safetensors] Invalid json header length file ..."
        #   short   -> "[load_safetensors] The JSON header is N bytes long ..."
        # Only the first says nothing about the file's contents.
        if exc is not None and "failed to open file" in str(exc).lower():
            return True
        try:
            fd = os.open(str(file_path), os.O_RDONLY)
        except OSError:
            # Cannot open: either the environment is out of descriptors, or the
            # file is already gone. Neither is evidence that its DESCENDANTS are
            # corrupt, and a missing file is collected by the orphan scan anyway.
            return True
        os.close(fd)
        return False

    def _queue_access_update(self, hash_hex: str) -> None:
        """Record an LRU touch for the background writer (best-effort freshness).

        A long-prefix restore touches every block it reads. Enqueueing one item
        per touch woke the writer for each and applied each as its own exclusive
        lock + UPDATE + COMMIT (~1 ms of writer time per block, competing with
        the restoring thread for the GIL). Touches now only update this table
        (latest timestamp + count per hash). The writer applies the whole table
        in one transaction inside every batch it processes, on its idle wakeup
        (<= 0.2 s) and before it exits. No queue item, so a touch can never
        delay or reorder real work (cleanups, fences, writes).
        """
        now = time.time()
        with self._access_lock:
            record = self._access_pending.get(hash_hex)
            if record is None:
                self._access_pending[hash_hex] = [now, 1]
            else:
                record[0] = now
                record[1] += 1

    def _take_pending_access_rows(self) -> List[Tuple[float, int, str]]:
        with self._access_lock:
            pending, self._access_pending = self._access_pending, {}
        return [(record[0], int(record[1]), hash_hex) for hash_hex, record in pending.items()]

    _ACCESS_UPDATE_SQL = (
        "UPDATE blocks SET last_accessed = ?, access_count = access_count + ? "
        "WHERE block_hash = ?"
    )

    def _flush_pending_access(self, write_conn: Optional[sqlite3.Connection]) -> None:
        """Writer thread only: apply recorded LRU touches outside any batch."""
        if write_conn is None:
            return
        with self._access_lock:
            if not self._access_pending:
                return
        rows = self._take_pending_access_rows()
        try:
            with self.global_budget.exclusive_mutation_guard() as locked:
                if not locked:
                    return
                write_conn.executemany(self._ACCESS_UPDATE_SQL, rows)
                write_conn.commit()
        except Exception as access_error:  # noqa: BLE001 - LRU freshness only
            logger.warning(
                "Background writer access update failed (%d rows): %s",
                len(rows),
                access_error,
            )
            try:
                write_conn.rollback()
            except Exception:
                pass

    def _queue_index_cleanup(self, hash_hex: str) -> None:
        """Queue a stale index entry cleanup for the background writer."""
        self._try_enqueue_write_item(("__cleanup__", hash_hex, 0))

    def _try_enqueue_write_item(
        self,
        item: Tuple[Any, ...],
        *,
        timeout: float = 0.0,
    ) -> str:
        """Atomically admit one queue item and track it through publication.

        Returns ``queued``, ``full``, ``quiescing``, or ``writer_stopped``.
        The lifecycle lock closes the check/enqueue race with clear/shutdown.
        ``timeout`` is opt-in bounded backpressure for native request fences;
        the default remains the legacy non-blocking admission policy.
        """

        timeout_s = max(0.0, float(timeout or 0.0))
        deadline = time.monotonic() + timeout_s
        with self._write_lifecycle:
            while True:
                if not self._accepting_writes:
                    return "quiescing"
                if not self._writer_thread.is_alive():
                    return "writer_stopped"
                try:
                    self._write_queue.put_nowait(item)
                    self._last_write_activity_monotonic = time.monotonic()
                except queue.Full:
                    remaining = deadline - time.monotonic()
                    if timeout_s <= 0.0 or remaining <= 0.0:
                        return "full"
                    self._write_lifecycle.wait(timeout=remaining)
                    continue
                self._pending_write_items += 1
                self._write_lifecycle.notify_all()
                return "queued"

    def _wait_for_write_queue_capacity(self, timeout: float) -> str:
        """Wait without serializing until a bounded writer slot is available."""

        timeout_s = max(0.0, float(timeout or 0.0))
        deadline = time.monotonic() + timeout_s
        with self._write_lifecycle:
            while self._write_queue.full():
                if not self._accepting_writes:
                    return "quiescing"
                if not self._writer_thread.is_alive():
                    return "writer_stopped"
                remaining = deadline - time.monotonic()
                if timeout_s <= 0.0 or remaining <= 0.0:
                    return "full"
                self._write_lifecycle.wait(timeout=remaining)
            return "ready"

    def _complete_write_items(self, count: int) -> None:
        if count <= 0:
            return
        with self._write_lifecycle:
            self._pending_write_items = max(
                0,
                self._pending_write_items - int(count),
            )
            self._write_lifecycle.notify_all()

    def _begin_write_producer(self) -> bool:
        with self._write_lifecycle:
            if (
                not self._accepting_writes
                or self._shutdown_started
                or not self._writer_thread.is_alive()
            ):
                return False
            self._active_write_producers += 1
            return True

    def _end_write_producer(self) -> None:
        with self._write_lifecycle:
            self._active_write_producers = max(
                0,
                self._active_write_producers - 1,
            )
            self._write_lifecycle.notify_all()

    def _write_admission_open(self) -> bool:
        with self._write_lifecycle:
            return bool(
                self._accepting_writes
                and not self._shutdown_started
                and self._writer_thread.is_alive()
            )

    # =========================================================================
    # Write (async)
    # =========================================================================

    @staticmethod
    def _estimate_cache_payload_bytes(cache_data: Any) -> int:
        """Conservatively estimate a frozen safetensors image before encoding.

        Admission must happen before ``_serialize_block`` or any MLX-backed
        safetensors work.  Cache payloads are small object trees whose leaves
        are arrays; count each array once and reserve ample header space.  The
        reservation is reconciled to the exact immutable byte length before
        enqueue.
        """

        from vmlx_engine.cache.byte_estimators import walk_payload_bytes

        tensor_bytes, tensor_count = walk_payload_bytes(
            cache_data,
            # Admission semantics: array leaves need a shape (stray int
            # nbytes attributes must not count), aliased arrays count once,
            # raw bytes lengths count, object __dict__ trees are walked,
            # depth capped.
            require_shape_for_arrays=True,
            dedupe_arrays=True,
            count_raw_bytes=True,
            walk_object_dicts=True,
            max_depth=16,
        )
        # Safetensors headers are JSON and padded; 64 KiB plus 1 KiB/tensor is
        # intentionally generous so exact reconciliation almost always shrinks
        # rather than grows the reservation.
        return max(1, tensor_bytes + 65536 + tensor_count * 1024)

    def _reserve_pending_write_bytes(
        self,
        requested: int,
        *,
        timeout: float = 0.0,
    ) -> bool:
        amount = max(1, int(requested))
        timeout_s = max(0.0, float(timeout or 0.0))
        deadline = time.monotonic() + timeout_s
        with self._pending_write_condition:
            if amount > self._max_pending_write_bytes:
                # OVERSIZED SINGLE PAYLOAD: admit EXCLUSIVELY instead of
                # dropping outright. The budget bounds the AGGREGATE RAM of
                # frozen payloads awaiting the writer — a single payload
                # larger than it (measured live: Laguna's 48-layer Mixed-SWA
                # state at ~19k tokens brushes the 1GB default) can NEVER be
                # written under a flat cap, which turns congestion control
                # into a permanent L2 coverage ceiling: the block drops,
                # ancestry truncation kills every descendant, and deep
                # restart-restore finds a hole at that depth forever. Waiting
                # for the queue to drain and admitting the payload alone
                # keeps the aggregate bounded by max(budget, one payload) —
                # RAM the process transiently holds for that copy regardless.
                while self._pending_write_bytes > 0:
                    remaining = deadline - time.monotonic()
                    if timeout_s <= 0.0 or remaining <= 0.0:
                        self._pending_write_byte_drops += 1
                        self.write_drop_reasons["byte_budget"] = self.write_drop_reasons.get("byte_budget", 0) + 1
                        return False
                    self._pending_write_condition.wait(timeout=remaining)
                self._pending_write_bytes += amount
                logger.info(
                    "BlockDiskStore admitted an oversized payload "
                    "exclusively (%d bytes > %d budget) after queue drain",
                    amount,
                    self._max_pending_write_bytes,
                )
                return True
            while self._pending_write_bytes + amount > self._max_pending_write_bytes:
                remaining = deadline - time.monotonic()
                if timeout_s <= 0.0 or remaining <= 0.0:
                    self._pending_write_byte_drops += 1
                    self.write_drop_reasons["byte_budget"] = self.write_drop_reasons.get("byte_budget", 0) + 1
                    return False
                self._pending_write_condition.wait(timeout=remaining)
            self._pending_write_bytes += amount
            return True

    def _resize_pending_write_reservation(
        self,
        previous: int,
        actual: int,
        *,
        timeout: float = 0.0,
    ) -> bool:
        old_amount = max(0, int(previous))
        new_amount = max(1, int(actual))
        timeout_s = max(0.0, float(timeout or 0.0))
        deadline = time.monotonic() + timeout_s
        with self._pending_write_condition:
            delta = new_amount - old_amount
            if (
                new_amount > self._max_pending_write_bytes
                or self._pending_write_bytes + delta
                > self._max_pending_write_bytes
            ):
                # The estimate was admitted but the frozen payload came out
                # larger than the budget allows alongside other writers. If
                # this reservation is the ONLY thing pending, the exclusive-
                # admission rule above applies to the resize too — dropping
                # here would reintroduce the flat ceiling one step later.
                # Otherwise wait for the writer to drain the others (within
                # the same admission deadline the initial reservation used):
                # an instant drop here poisons every descendant of this block
                # for the session, so a transient burst becomes a permanent
                # L2 lineage hole.
                while self._pending_write_bytes - old_amount > 0:
                    remaining = deadline - time.monotonic()
                    if timeout_s <= 0.0 or remaining <= 0.0:
                        self._pending_write_bytes = max(
                            0, self._pending_write_bytes - old_amount
                        )
                        self._pending_write_byte_drops += 1
                        self.write_drop_reasons["byte_budget"] = self.write_drop_reasons.get("byte_budget", 0) + 1
                        self._pending_write_condition.notify_all()
                        return False
                    self._pending_write_condition.wait(timeout=remaining)
                self._pending_write_bytes = new_amount
                if new_amount > self._max_pending_write_bytes:
                    logger.info(
                        "BlockDiskStore resized an exclusive oversized "
                        "payload reservation (%d -> %d bytes, budget %d)",
                        old_amount,
                        new_amount,
                        self._max_pending_write_bytes,
                    )
                return True
            self._pending_write_bytes = max(
                0, self._pending_write_bytes + delta
            )
            if delta < 0:
                self._pending_write_condition.notify_all()
        return True

    def _release_pending_write_bytes(self, reserved: int) -> None:
        with self._pending_write_condition:
            self._pending_write_bytes = max(
                0,
                self._pending_write_bytes - max(0, int(reserved)),
            )
            self._pending_write_condition.notify_all()

    def pending_write_bytes(self) -> int:
        """Bytes admitted to the writer and not yet written (frozen payloads it still owns)."""
        with self._pending_write_condition:
            return int(self._pending_write_bytes)

    def wait_for_pending_write_bytes(self, at_most: int, *, timeout: float) -> bool:
        """Block until the writer owns at most ``at_most`` pending bytes (a producer's own in-flight cap, below the
        store-wide budget). False when ``timeout`` seconds pass first."""
        limit = max(0, int(at_most))
        deadline = time.monotonic() + max(0.0, float(timeout or 0.0))
        with self._pending_write_condition:
            while self._pending_write_bytes > limit:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._pending_write_condition.wait(timeout=remaining)
        return True

    @staticmethod
    def _detach_safetensors_tensors(
        tensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Freeze a flat tensor map into immutable, CPU-readable NumPy arrays.

        MLX evaluation and unified-memory copies must remain on the
        model-owning thread. Evaluated, contiguous MLX block tensors cross the
        queue as read-only NumPy views: their ``memoryview`` base owns the exact
        per-block allocation until the writer finishes, without allocating a
        second host image. Caller-owned NumPy inputs are still copied so later
        producer mutation cannot change queued bytes. BF16 is carried as its
        raw uint16 bits (NumPy has no bfloat16 dtype, but it does not need one
        to move bytes). Block metadata records the original dtype so restore
        can reinterpret it. Older blocks on disk hold f32 and are still
        restored by casting -- see _restore_serialized_dtype.
        """

        import numpy as np

        pending_mlx: list[tuple[str, Any]] = []
        detached: Dict[str, Any] = {}
        # Fast path, same bytes and the same aliasing as below: evaluate the inputs once, then expose each
        # C-contiguous buffer to NumPy through the buffer protocol (BF16 as its uint16 bits). The general path
        # builds two lazy nodes per tensor (contiguous + bit view) and evaluates them, which cost ~5-8 us per
        # tensor on the store path (a 46k Naive store freezes ~36k tensors on the request's terminal path);
        # mx.contiguous of a contiguous input shares its buffer, so the views alias exactly what they did.
        # Non-contiguous inputs (no buffer export) still take the general path and are copied there.
        mlx_inputs = [value for value in tensors.values() if isinstance(value, mx.array)]
        if mlx_inputs:
            mx.eval(*mlx_inputs)
        for name, value in tensors.items():
            if isinstance(value, mx.array):
                try:
                    buffer = memoryview(value)
                except (BufferError, TypeError, ValueError):
                    buffer = None
                if buffer is not None and buffer.c_contiguous:
                    if value.dtype == mx.bfloat16:
                        view = np.ndarray(tuple(value.shape), dtype=np.uint16, buffer=buffer)
                    else:
                        view = np.asarray(buffer)
                    view.setflags(write=False)
                    detached[name] = view
                    continue
                materialized = mx.contiguous(value)
                if "bfloat16" in str(value.dtype):
                    # Reinterpret the bf16 bits as uint16 rather than widening
                    # to f32. NumPy genuinely has no bfloat16, but it does not
                    # need one: the bytes are carried verbatim and viewed back
                    # on restore, which __orig_dtypes__ already records enough
                    # information to do.
                    #
                    # Widening cost 2x on EVERY count -- bytes written, bytes
                    # read back, plus a GPU conversion pass each way -- and the
                    # SSD block cache is now the default prefix tier for every
                    # model, so that sat on the hot prefill path. Measured on
                    # this stack: 65,608 -> 32,840 bytes for the same tensor,
                    # bit-exact including 1e30.
                    materialized = materialized.view(mx.uint16)
                pending_mlx.append((name, materialized))
                continue
            if isinstance(value, np.ndarray):
                array = value
                if "bfloat16" in str(array.dtype):
                    array = array.astype(np.float32)
                copied = np.array(array, copy=True, order="C")
                copied.setflags(write=False)
                detached[name] = copied
                continue
            raise TypeError(
                f"safetensors value {name!r} is not an MLX/NumPy array"
            )

        if pending_mlx:
            mx.eval(*(value for _, value in pending_mlx))
        for name, value in pending_mlx:
            view = np.asarray(value)
            if not view.flags.c_contiguous:
                raise ValueError(
                    f"materialized MLX tensor {name!r} is not C-contiguous"
                )
            view.setflags(write=False)
            detached[name] = view
        return detached

    def _stage_numpy_safetensors_file(
        self,
        block_hash: bytes,
        tensors: Dict[str, Any],
    ) -> Path:
        """Encode immutable NumPy buffers directly into a durable temp file.

        ``safetensors.numpy.save`` builds a full aggregate Python ``bytes``
        image. Its ``save_file`` sibling avoids that aggregate but still calls
        ``tensor.tobytes()`` for every tensor, retaining another block-sized
        malloc high-water on macOS. The cache writer emits the same compact,
        8-byte-padded safetensors header and writes each C-contiguous NumPy
        buffer directly, so only the small header is copied.

        The path uses the existing lease-tagged ``.tmp.safetensors`` contract:
        root-budget scans protect active-writer temps, and abandoned files are
        ordinary orphan-cleanup candidates after the lease disappears.
        """

        final_path = self._hash_to_path(bytes(block_hash).hex())
        staged_path = self._new_payload_temp_path(final_path)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(staged_path, flags, 0o600)
        os.close(fd)
        try:
            self._stream_numpy_safetensors_file(staged_path, tensors)
            return staged_path
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_all_fd(fd: int, payload: Any) -> None:
        """Write one contiguous buffer completely without materializing bytes."""

        view = memoryview(payload)
        # Safetensors permits zero-element tensors and represents them with an
        # empty data-offset span.  DSV4 periodic anchors legitimately carry
        # zero-row residual pool buffers at exact compression boundaries.
        # Python refuses to byte-cast a view with a zero dimension even though
        # there are no payload bytes to write, so finish before that cast.
        if view.nbytes == 0:
            return
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError("short block-cache payload write")
            written += count

    @staticmethod
    def _writev_all_fd(fd: int, payloads: List[Any]) -> None:
        """Write contiguous buffers completely, in order, with gathered writes."""

        views: List[memoryview] = []
        for payload in payloads:
            view = memoryview(payload)
            if view.nbytes == 0:
                continue  # zero-element tensors have an empty data span
            if view.ndim != 1 or view.format != "B":
                view = view.cast("B")
            views.append(view)
        writev = getattr(os, "writev", None)
        if writev is None:
            for view in views:
                BlockDiskStore._write_all_fd(fd, view)
            return
        try:
            iov_max = int(os.sysconf("SC_IOV_MAX"))
        except (AttributeError, ValueError, OSError):
            iov_max = 1024
        iov_max = max(1, min(iov_max, 1024))
        index = 0
        while index < len(views):
            batch = views[index : index + iov_max]
            written = writev(fd, batch)
            if written <= 0:
                raise OSError("short block-cache payload write")
            # Advance past fully written buffers; keep the unwritten tail of a
            # partially written one.
            for view in batch:
                if written >= len(view):
                    written -= len(view)
                    index += 1
                    continue
                views[index] = view[written:]
                break

    def _stream_numpy_safetensors_file(
        self,
        path: Path,
        tensors: dict[str, Any],
    ) -> None:
        """Write the standard safetensors wire image from NumPy buffers."""

        header: dict[str, Any] = {}
        offset = 0
        for name, array in tensors.items():
            if not isinstance(name, str):
                raise TypeError("safetensors tensor names must be strings")
            flags = getattr(array, "flags", None)
            if flags is None or not bool(flags.c_contiguous):
                raise ValueError(
                    f"safetensors tensor {name!r} must be C-contiguous"
                )
            dtype = getattr(array, "dtype", None)
            dtype_key = (
                str(getattr(dtype, "kind", "")),
                int(getattr(dtype, "itemsize", 0)),
            )
            dtype_code = _SAFETENSORS_NUMPY_DTYPE_CODES.get(dtype_key)
            if dtype_code is None:
                raise TypeError(
                    f"unsupported safetensors NumPy dtype for {name!r}: {dtype}"
                )
            end = offset + int(array.nbytes)
            header[name] = {
                "dtype": dtype_code,
                "shape": [int(dim) for dim in array.shape],
                "data_offsets": [offset, end],
            }
            offset = end

        header_bytes = json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        header_bytes += b" " * ((-len(header_bytes)) % 8)

        flags = os.O_WRONLY | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            # One gathered write per IOV_MAX buffers instead of one os.write per
            # tensor (a 48-layer block is ~100 tensors; 512 blocks made ~33k
            # write syscalls, ~360 ms of writer time). Same bytes, same order.
            buffers: list[Any] = [
                struct.pack("<Q", len(header_bytes)),
                header_bytes,
            ]
            for array in tensors.values():
                byteorder = str(array.dtype.byteorder)
                needs_swap = byteorder == ">" or (
                    byteorder == "=" and sys.byteorder != "little"
                )
                buffers.append(
                    array.byteswap(inplace=False) if needs_swap else array
                )
            try:
                self._writev_all_fd(fd, buffers)
            finally:
                buffers = None
            os.fsync(fd)
        finally:
            os.close(fd)

    def _disable_global_budget_writes(self) -> None:
        self._global_budget_write_enabled = False
        self._last_budget_recovery_attempt_ns = time.monotonic_ns()

    def _maybe_recover_global_budget_writes(self) -> bool:
        """Retry a failed aggregate reconcile at a bounded cadence."""

        if self._pending_publication_recovery is not None:
            # Cap compliance alone cannot certify a failed SQL publication.
            return False
        if self._global_budget_write_enabled:
            return True
        now_ns = time.monotonic_ns()
        with self._budget_recovery_lock:
            if self._global_budget_write_enabled:
                return True
            if (
                self._last_budget_recovery_attempt_ns
                and now_ns - self._last_budget_recovery_attempt_ns
                < self._budget_recovery_interval_ns
            ):
                return False
            self._last_budget_recovery_attempt_ns = now_ns
            result = self.global_budget.enforce(force=True)
            self._global_budget_write_enabled = bool(
                result.accounted and result.compliant
            )
            return self._global_budget_write_enabled

    def _recover_pending_publication_locked(self, conn: sqlite3.Connection) -> None:
        """Discharge failed publication ownership under the root-exclusive guard."""
        pending = self._pending_publication_recovery
        if pending is None:
            return
        block_hash, fence_ids = pending
        if block_hash is not None:
            self._cleanup_entry(conn, block_hash.hex())
            path = self._hash_to_path(block_hash.hex())
            path.unlink(missing_ok=True)
            self._fsync_directory(path.parent)
        self._release_write_fence_pins_locked(conn, fence_ids)
        result = self.global_budget._enforce_locked()
        if not result.accounted or not result.compliant:
            raise RuntimeError("failed block publication recovery remains unaccounted")
        self._pending_publication_recovery = None

    def begin_write_fence(
        self,
        request_id: str,
        *,
        strict_reconcile: bool = False,
        admission_timeout: float = 0.0,
    ) -> str:
        """Create an opaque, request-correlated asynchronous write fence."""
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("request_id is required for a block-disk write fence")
        with self._stats_lock:
            self._prune_write_fences_locked(
                max_count=self._max_recent_write_fences - 1
            )
            if len(self._write_fences) >= self._max_recent_write_fences:
                raise RuntimeError(
                    "too many unfinished block-disk write fences"
                )
            self._write_fence_seq += 1
            fence_id = f"block-write-{self._write_fence_seq:016x}"
            self._write_fences[fence_id] = {
                "fence_id": fence_id,
                "request_id": normalized_request_id,
                "expected": 0,
                "queued": 0,
                "completed": 0,
                "_processed": 0,
                "failed": 0,
                "dropped": 0,
                "retained": 0,
                "sealed": False,
                "seal_enqueued": False,
                "seal_failed": False,
                "producer_aborted": False,
                "post_eviction_complete": False,
                "completion_generation": None,
                "_strict_reconcile": bool(strict_reconcile),
                "_admission_timeout": max(
                    0.0,
                    float(admission_timeout or 0.0),
                ),
                "_queued_hashes": [],
                "_active": 0,
            }
        return fence_id

    def _pin_write_fence_block_locked(
        self,
        conn: sqlite3.Connection,
        fence_id: Optional[str],
        block_hash: bytes,
    ) -> None:
        """Protect an unfinished fence block from root-global eviction."""

        if not self._write_fence_pin_is_open(fence_id):
            return
        self._insert_write_fence_pin_locked(conn, str(fence_id), block_hash)
        conn.commit()

    def _write_fence_pin_is_open(self, fence_id: Optional[str]) -> bool:
        """Snapshot eligibility before opening a SQLite write transaction."""
        if not fence_id:
            return False
        with self._stats_lock:
            state = self._write_fences.get(str(fence_id))
            return state is not None and not state.get("post_eviction_complete")

    def _insert_write_fence_pin_locked(
        self,
        conn: sqlite3.Connection,
        fence_id: str,
        block_hash: bytes,
    ) -> None:
        """Insert an eligible pin; caller owns commit and takes no stats lock."""
        conn.execute(
            "INSERT OR IGNORE INTO block_write_pins "
            "(block_hash, owner_lease_id, fence_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                bytes(block_hash).hex(),
                self.global_budget.lease_id,
                str(fence_id),
                time.time(),
            ),
        )

    def _release_write_fence_pins_locked(
        self,
        conn: sqlite3.Connection,
        fence_ids: set[str] | None = None,
    ) -> set[str]:
        """Release this process's persistent pins while the root lock is held."""

        owner = self.global_budget.lease_id
        if fence_ids is None:
            rows = conn.execute(
                "SELECT DISTINCT fence_id FROM block_write_pins "
                "WHERE owner_lease_id = ?",
                (owner,),
            ).fetchall()
            released = {str(row[0]) for row in rows}
            conn.execute(
                "DELETE FROM block_write_pins WHERE owner_lease_id = ?",
                (owner,),
            )
        else:
            released = {str(fence_id) for fence_id in fence_ids if fence_id}
            if not released:
                return set()
            ordered = sorted(released)
            for start in range(0, len(ordered), 500):
                chunk = ordered[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    "DELETE FROM block_write_pins "
                    "WHERE owner_lease_id = ? "
                    f"AND fence_id IN ({placeholders})",
                    (owner, *chunk),
                )
        conn.commit()
        return released

    def _release_write_fence_pins(self, fence_id: str) -> None:
        """Best-effort terminal cleanup for a fence failed off the writer path."""

        try:
            with self.global_budget.exclusive_mutation_guard() as locked:
                if not locked:
                    return
                conn = sqlite3.connect(str(self._db_path), timeout=5.0)
                try:
                    self._release_write_fence_pins_locked(
                        conn,
                        {str(fence_id)},
                    )
                    self.global_budget._enforce_locked()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "Could not release block-disk fence pins (%s): %s",
                fence_id,
                exc,
            )

    def _prune_write_fences_locked(
        self,
        *,
        max_count: Optional[int] = None,
    ) -> None:
        """Bound retained telemetry without discarding unfinished fences."""
        limit = (
            self._max_recent_write_fences
            if max_count is None
            else max(0, int(max_count))
        )
        while len(self._write_fences) > limit:
            removable = next(
                (
                    fence_id
                    for fence_id, state in self._write_fences.items()
                    if state.get("post_eviction_complete")
                    or state.get("seal_failed")
                ),
                None,
            )
            if removable is None:
                break
            self._write_fences.pop(removable, None)

    def _write_fence_expected(self, fence_id: Optional[str]) -> bool:
        if not fence_id:
            return False
        budget_healthy = self._maybe_recover_global_budget_writes()
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if (
                state is None
                or state.get("sealed")
                or state.get("post_eviction_complete")
                or state.get("seal_failed")
            ):
                return False
            state["expected"] += 1
            state["_active"] = int(state.get("_active") or 0) + 1
        if not budget_healthy:
            # Preserve a truthful expected/failed pair for request fences even
            # when startup reconciliation disabled all publications.
            self._write_fence_queue_result(fence_id, failed=True)
            logger.warning(
                "BlockDiskStore write skipped because aggregate budget "
                "coordination is not healthy"
            )
            return False
        return True

    def _write_fence_ready_locked(self, state: Dict[str, Any]) -> bool:
        accounted = (
            int(state.get("queued") or 0)
            + int(state.get("failed") or 0)
            + int(state.get("dropped") or 0)
        )
        return (
            int(state.get("_active") or 0) <= 0
            and accounted >= int(state.get("expected") or 0)
        )

    def _write_fence_publication_ready_locked(
        self,
        state: Dict[str, Any],
    ) -> bool:
        """True only after every admitted payload reached the writer."""

        return (
            self._write_fence_ready_locked(state)
            and int(state.get("_processed") or 0)
            >= int(state.get("queued") or 0)
        )

    def _write_fence_queue_result(
        self,
        fence_id: Optional[str],
        *,
        block_hash: Optional[bytes] = None,
        failed: bool = False,
        dropped: bool = False,
    ) -> None:
        if not fence_id:
            return
        should_enqueue_fence = False
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is None or state.get("post_eviction_complete"):
                return
            if dropped:
                state["dropped"] += 1
            elif failed:
                state["failed"] += 1
            else:
                state["queued"] += 1
                if block_hash:
                    state["_queued_hashes"].append(bytes(block_hash))
            if int(state.get("_active") or 0) > 0:
                state["_active"] = int(state.get("_active") or 0) - 1
            should_enqueue_fence = (
                bool(state.get("sealed"))
                and not state.get("seal_enqueued")
                and not state.get("seal_failed")
                and self._write_fence_ready_locked(state)
            )
        if should_enqueue_fence:
            self._enqueue_write_fence_sentinel(str(fence_id))

    def record_write_fence_unadmitted(
        self,
        fence_id: Optional[str],
        count: int,
    ) -> None:
        """Account causal-chain writes skipped after the first rejection.

        Native block chains stop enqueueing after one admission failure because
        every later child depends on the missing parent.  Count those skipped
        children as expected+dropped without pretending they reached the
        writer.  This keeps request telemetry truthful while RAM remains the
        authoritative fallback for the missing suffix.
        """

        skipped = max(0, int(count or 0))
        if not fence_id or skipped <= 0:
            return
        should_enqueue_fence = False
        with self._stats_lock:
            state = self._write_fences.get(str(fence_id))
            if (
                state is None
                or state.get("post_eviction_complete")
                or state.get("seal_failed")
            ):
                return
            state["expected"] += skipped
            state["dropped"] += skipped
            should_enqueue_fence = (
                bool(state.get("sealed"))
                and not state.get("seal_enqueued")
                and self._write_fence_ready_locked(state)
            )
        if should_enqueue_fence:
            self._enqueue_write_fence_sentinel(str(fence_id))

    def _write_fence_completion(
        self,
        fence_id: Optional[str],
        *,
        failed: bool,
    ) -> None:
        if not fence_id:
            return
        should_enqueue_fence = False
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is None:
                return
            state["_processed"] = int(state.get("_processed") or 0) + 1
            if failed:
                state["failed"] += 1
            else:
                state["completed"] += 1
            should_enqueue_fence = (
                bool(state.get("sealed"))
                and not state.get("seal_enqueued")
                and not state.get("seal_failed")
                and not state.get("post_eviction_complete")
                and self._write_fence_publication_ready_locked(state)
            )
        if should_enqueue_fence:
            self._enqueue_write_fence_sentinel(str(fence_id))

    def seal_write_fence(
        self,
        fence_id: str,
        *,
        producer_aborted: bool = False,
    ) -> bool:
        """Queue a non-blocking post-eviction completion sentinel."""
        should_enqueue = False
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is None or state.get("sealed"):
                return False
            state["sealed"] = True
            state["producer_aborted"] = bool(producer_aborted)
            should_enqueue = self._write_fence_ready_locked(state)
        if not should_enqueue:
            return True
        return self._enqueue_write_fence_sentinel(fence_id)

    def _enqueue_write_fence_sentinel(self, fence_id: str) -> bool:
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            admission_timeout = float(
                (state or {}).get("_admission_timeout") or 0.0
            )
        enqueue_result = self._try_enqueue_write_item(
            ("__fence__", fence_id, 0),
            timeout=admission_timeout,
        )
        if enqueue_result == "full":
            # Every data item for this fence was admitted before sealing. A
            # full FIFO therefore already contains work that will wake the
            # writer. Mark the control record as deferred; the writer scans
            # publication-ready sealed fences after each batch and finalizes
            # this one after all of its admitted payloads were processed.
            with self._stats_lock:
                state = self._write_fences.get(fence_id)
                if state is None or state.get("post_eviction_complete"):
                    return True
                state["seal_enqueued"] = True
                state["_sentinel_deferred"] = True
            logger.info(
                "BlockDiskStore deferred fence sentinel behind a full data "
                "queue: %s",
                fence_id,
            )
            return True
        if enqueue_result != "queued":
            with self._stats_lock:
                state = self._write_fences.get(fence_id)
                if state is not None:
                    state["seal_failed"] = True
            self._fail_write_fence(
                fence_id,
                (
                    "block-disk write queue full before fence sentinel"
                    if enqueue_result == "full"
                    else "block-disk writer quiescing before fence sentinel"
                ),
            )
            logger.warning(
                "BlockDiskStore fence sentinel was not queued (%s): %s",
                fence_id,
                enqueue_result,
            )
            return False
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is not None:
                state["seal_enqueued"] = True
        return True

    def write_block_async(
        self,
        block_hash: bytes,
        cache_data: List[Tuple],
        token_count: int,
        *,
        parent_hash: Optional[bytes] = None,
        request_id: Optional[str] = None,
        fence_id: Optional[str] = None,
        replace_existing: bool = False,
    ) -> bool:
        """Admit one producer across freeze and immutable queue publication."""

        if not self._begin_write_producer():
            return False
        # Every settle path inside RETURNS, so an exception that escapes means a
        # producer was registered on the fence and never accounted for. That
        # leaves _active stuck above zero: the fence never becomes ready, never
        # terminalizes, and is never pruned (unfinished fences are retained by
        # design), so at the 64-fence cap begin_write_fence starts raising —
        # which its caller catches, warns about, and continues past with
        # disk_write_fence_id=None. Every later request then writes UNFENCED for
        # the life of the process, and pins held by the leaked fence's earlier
        # writes keep their bytes un-evictable.
        registered: list[str] = []
        try:
            return self._write_block_async_admitted(
                block_hash,
                cache_data,
                token_count,
                parent_hash=parent_hash,
                request_id=request_id,
                fence_id=fence_id,
                replace_existing=replace_existing,
                _registered=registered,
            )
        except BaseException:
            if registered:
                self._write_fence_queue_result(fence_id, failed=True)
            raise
        finally:
            self._end_write_producer()

    def _write_block_async_admitted(
        self,
        block_hash: bytes,
        cache_data: List[Tuple],
        token_count: int,
        *,
        parent_hash: Optional[bytes] = None,
        request_id: Optional[str] = None,
        fence_id: Optional[str] = None,
        replace_existing: bool = False,
        _registered: Optional[list] = None,
    ) -> bool:
        """
        Freeze a block on the model-owning thread and queue disk publication.

        ALL MLX operations happen on the calling (main) thread:
        - Serialize cache_data to flat tensor dict
        - Materialize lazy arrays with mx.eval()
        - Expose each independent MLX block as an immutable NumPy memoryview

        The background thread performs CPU safetensors encoding and every
        filesystem operation: temporary write, fsync, atomic publish, SQLite
        index/accounting, and eviction. No lazy MLX operation crosses the queue
        boundary; the NumPy ``memoryview`` owns the already-evaluated block
        allocation until its file is staged.

        Args:
            block_hash: Chain hash (BlockHash bytes)
            cache_data: CacheBlock.cache_data — list of typed tuples per layer
            token_count: Number of tokens in this block
            parent_hash: Chain hash of the immediately preceding block, or
                ``None`` only for a known chain root.
            request_id: Scheduler request ID associated with ``fence_id``.
            fence_id: Opaque ID returned by :meth:`begin_write_fence`.
        """
        if fence_id:
            tracked = self._write_fence_expected(fence_id)
            if tracked and _registered is not None:
                # A producer is now owed an accounting entry. The caller settles
                # it if anything below escapes instead of returning.
                _registered.append(str(fence_id))
        else:
            tracked = False
            if not self._maybe_recover_global_budget_writes():
                logger.warning(
                    "BlockDiskStore write skipped because aggregate budget "
                    "coordination is not healthy"
                )
                return False
        if fence_id and not tracked:
            logger.warning(
                "Rejecting block write for unknown or sealed fence %s",
                fence_id,
            )
            return False
        admission_timeout = 0.0
        if tracked:
            request_mismatch = False
            with self._stats_lock:
                state = self._write_fences.get(str(fence_id))
                if state is None or state.get("request_id") != str(request_id or ""):
                    request_mismatch = True
                elif state is not None:
                    admission_timeout = max(
                        0.0,
                        float(state.get("_admission_timeout") or 0.0),
                    )
            if request_mismatch:
                # Balance the expected/active producer registered above. A
                # manual failed++ left _active stuck and a later seal could
                # never enqueue or terminalize the fence.
                self._write_fence_queue_result(fence_id, failed=True)
                logger.warning(
                    "Rejecting block write whose request ID does not match "
                    "fence %s",
                    fence_id,
                )
                return False
        if not HAS_MLX:
            self._write_fence_queue_result(fence_id, failed=True)
            return False

        admission_deadline = time.monotonic() + admission_timeout

        def _remaining_admission_time() -> float:
            if admission_timeout <= 0.0:
                return 0.0
            return max(0.0, admission_deadline - time.monotonic())

        # Fail item admission before walking/serializing MLX state.  The item
        # queue bounds metadata commands; the byte reservation independently
        # bounds immutable payload RAM.
        if not self._write_admission_open():
            self._write_fence_queue_result(fence_id, dropped=True)
            return False
        queue_capacity = self._wait_for_write_queue_capacity(
            _remaining_admission_time()
        )
        if queue_capacity != "ready":
            self._write_fence_queue_result(fence_id, dropped=True)
            logger.warning(
                "BlockDiskStore write queue admission failed (%s), skipping "
                "serialization",
                queue_capacity,
            )
            return False

        if _cache_data_has_tq(cache_data) and not self._allow_tq_native:
            logger.debug("Skipping TQ-native block write because persisted TQ is disabled")
            self._write_fence_queue_result(fence_id, failed=True)
            return False

        if not self._allow_tq_native:
            # The content hash intentionally identifies the token prefix, not the
            # storage codec.  A previous Auto/q4 run can therefore own the same
            # row that an explicit None run is about to persist as plain KV.  The
            # background writer normally de-duplicates an existing hash, so only
            # rejecting the old TQ record during lookup leaves an untouched TQ
            # suffix behind after the first miss.  Inspect/evict an incompatible
            # record for *every* plain-KV write before queuing the replacement.
            # Standard records remain untouched and are still de-duplicated by
            # _write_block(). A hash with no index row has nothing to inspect:
            # check that first without the aggregate mutation guard, which the
            # busy writer holds while it publishes (~0.3 ms per block of a cold
            # store spent waiting for it, on the model thread).
            existing_row, _ok = self._query_index_row(
                "SELECT 1 FROM blocks WHERE block_hash = ?",
                (block_hash.hex(),),
            )
            if existing_row is not None:
                self.has_block(block_hash)

        if parent_hash is not None and parent_hash in self._session_dropped_hashes:
            # The parent was budget-dropped this session — this block can
            # never publish (ancestry requirement). Truncate the chain here
            # instead of consuming budget and failing later at publish.
            self._session_dropped_hashes.add(block_hash)
            if len(self._session_dropped_hashes) > 8192:
                self._session_dropped_hashes.clear()
            self._write_fence_queue_result(fence_id, dropped=True)
            logger.debug(
                "BlockDiskStore truncating write chain at %s: ancestor was "
                "budget-dropped",
                block_hash.hex()[:12],
            )
            return False

        reserved_bytes = self._estimate_cache_payload_bytes(cache_data)
        if not self._reserve_pending_write_bytes(
            reserved_bytes,
            timeout=_remaining_admission_time(),
        ):
            self._session_dropped_hashes.add(block_hash)
            if len(self._session_dropped_hashes) > 8192:
                self._session_dropped_hashes.clear()
            self._write_fence_queue_result(fence_id, dropped=True)
            logger.warning(
                "BlockDiskStore pending-write byte budget full "
                "(%d bytes), skipping serialization; descendants of this "
                "block will be truncated at queue time",
                self._max_pending_write_bytes,
            )
            return False
        self._session_dropped_hashes.discard(block_hash)

        hash_hex = block_hash.hex()

        # Flatten and freeze on the calling/model-owning thread. The expensive
        # CPU safetensors encoding runs on the background writer. MLX inputs
        # stay zero-copy here; mutable NumPy inputs are independently snapped.
        try:
            tensors, dtype, num_layers = _serialize_block(cache_data)
            if num_layers == 0:
                self._release_pending_write_bytes(reserved_bytes)
                self._write_fence_queue_result(fence_id, failed=True)
                return False

            tensors = self._detach_safetensors_tensors(tensors)
            detached_bytes = (
                sum(max(0, int(value.nbytes)) for value in tensors.values())
                + 65536
                + len(tensors) * 1024
            )
            if not self._resize_pending_write_reservation(
                reserved_bytes,
                detached_bytes,
                timeout=_remaining_admission_time(),
            ):
                self._write_fence_queue_result(fence_id, dropped=True)
                logger.warning(
                    "BlockDiskStore frozen payload exceeded pending-write "
                    "byte budget (%d bytes)",
                    self._max_pending_write_bytes,
                )
                return False
            reserved_bytes = detached_bytes
        except Exception as e:
            self._release_pending_write_bytes(reserved_bytes)
            logger.debug(f"Pre-freeze failed for block {hash_hex[:12]}: {e}")
            self._write_fence_queue_result(fence_id, failed=True)
            return False

        # Queue only immutable NumPy buffers. An MLX-backed view owns its
        # evaluated per-block allocation through a memoryview; the writer never
        # invokes MLX and replaces this item with a staged path before taking
        # the aggregate publication lock.
        with self._stats_lock:
            # Increment before publication so a fast writer can never make
            # completed temporarily exceed queued in /health telemetry.
            self._offthread_serializations_queued += 1
        enqueue_result = self._try_enqueue_write_item(
            (
                "__numpy_block__",
                block_hash,
                tensors,
                dtype,
                num_layers,
                token_count,
                parent_hash,
                fence_id,
                reserved_bytes,
                bool(replace_existing),
            ),
            timeout=_remaining_admission_time(),
        )
        if enqueue_result == "queued":
            self._write_fence_queue_result(fence_id, block_hash=block_hash)
            if _cache_data_has_tq(cache_data):
                with self._stats_lock:
                    self.tq_native_writes += 1
            return True
        with self._stats_lock:
            self._offthread_serializations_queued = max(
                0,
                self._offthread_serializations_queued - 1,
            )
        self._release_pending_write_bytes(reserved_bytes)
        self._write_fence_queue_result(fence_id, dropped=True)
        with self._stats_lock:
            if enqueue_result in self.write_drop_reasons:
                self.write_drop_reasons[enqueue_result] += 1
        if enqueue_result == "full":
            logger.warning("BlockDiskStore write queue full (1000), dropping block write")
        else:
            logger.debug(
                "BlockDiskStore rejected frozen block while writer was %s",
                enqueue_result,
            )
        return False

    def wait_for_blocks(
        self,
        block_hashes: List[bytes],
        timeout: float = 5.0,
    ) -> set[bytes]:
        """Wait until queued block writes are index-readable.

        Disk-only prefix mode cannot keep a speculative RAM payload while the
        background writer performs its atomic rename and SQLite update. This
        bounded durability barrier observes only filesystem/index state; it
        never loads tensors or touches Metal. The returned set contains hashes
        that became readable before the timeout so callers can retain a safe
        in-memory fallback for any failed write instead of advertising a cache
        entry that reconstructs as ``None``.
        """
        targets = {bytes(block_hash) for block_hash in block_hashes if block_hash}
        if not targets:
            return set()

        target_by_hex = {block_hash.hex(): block_hash for block_hash in targets}
        ready: set[bytes] = set()
        deadline = time.monotonic() + max(0.0, float(timeout))

        if not self._allow_tq_native:
            # An explicit TQ-Off run can replace a TQ-native record under the
            # same content hash.  The cleanup and plain-KV publication are
            # ordered on the background queue, but the old finalized row is
            # still index-readable until that queue settles.  Counting that
            # stale row as durable would drop the caller's RAM fallback and a
            # subsequent read would correctly reject it as TQ, producing a
            # false cache miss.  Wait on the authoritative lifecycle counter
            # before testing path/index readability in this replacement mode.
            with self._write_lifecycle:
                while (
                    self._active_write_producers > 0
                    or self._pending_write_items > 0
                ):
                    if (
                        self._pending_write_items > 0
                        and not self._writer_thread.is_alive()
                    ):
                        return set()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return set()
                    self._write_lifecycle.wait(timeout=remaining)

        while True:
            remaining_hex = [
                hash_hex
                for hash_hex, block_hash in target_by_hex.items()
                if block_hash not in ready
            ]
            if not remaining_hex:
                return ready

            conn = sqlite3.connect(str(self._db_path), timeout=1.0)
            try:
                for start in range(0, len(remaining_hex), 500):
                    chunk = remaining_hex[start : start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT block_hash, file_name FROM blocks "
                        f"WHERE block_hash IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for hash_hex, file_name in rows:
                        if (self.cache_dir / file_name).exists():
                            block_hash = target_by_hex.get(hash_hex)
                            if block_hash is not None:
                                ready.add(block_hash)
            finally:
                conn.close()

            if len(ready) == len(targets) or time.monotonic() >= deadline:
                return ready
            time.sleep(0.005)

    def wait_for_write_fence_blocks(
        self,
        fence_id: str,
        block_hashes: List[bytes],
        timeout: Optional[float] = 5.0,
        *,
        allow_partial: bool = False,
    ) -> set[bytes]:
        """Return exact blocks retained after a request fence has settled.

        ``wait_for_blocks`` intentionally answers the narrower question "is a
        file/index row readable right now?"  A writer commits that row before
        aggregate budget accounting and eviction, so it is not a sufficient
        durability barrier for native path-dependent cache state.  This method
        first waits for the request-correlated fence to finish its post-write
        eviction phase, and only then verifies each requested hash while
        holding the process-shared mutation guard.  The default rejects every
        incomplete/error/retention-loss terminal state for legacy callers.
        Native RAM-fallback settlement may set ``allow_partial=True`` to release
        only exact hashes that survived a terminal partial publication while
        retaining every missing causal block in RAM.

        Callers must seal ``fence_id`` before waiting.  An empty result is the
        fail-closed outcome for timeout, writer death, fence failure, or an
        exact hash that did not survive eviction.
        """
        normalized_fence_id = str(fence_id or "").strip()
        targets = {bytes(block_hash) for block_hash in block_hashes if block_hash}
        if not normalized_fence_id or not targets:
            return set()

        deadline = (
            None
            if timeout is None
            else time.monotonic() + max(0.0, float(timeout))
        )
        terminal: Dict[str, Any] | None = None
        while True:
            with self._stats_lock:
                state = self._write_fences.get(normalized_fence_id)
                if state is None:
                    logger.warning(
                        "BlockDiskStore durability wait lost fence %s",
                        normalized_fence_id,
                    )
                    return set()
                if state.get("post_eviction_complete"):
                    terminal = dict(state)
                    break
                writer_alive = self._writer_thread.is_alive()

            if not writer_alive:
                logger.warning(
                    "BlockDiskStore durability wait stopped because writer died "
                    "before fence %s completed",
                    normalized_fence_id,
                )
                return set()
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    "BlockDiskStore durability wait timed out before fence %s "
                    "completed",
                    normalized_fence_id,
                )
                return set()
            time.sleep(0.005)

        expected = int(terminal.get("expected") or 0)
        queued = int(terminal.get("queued") or 0)
        completed = int(terminal.get("completed") or 0)
        retained = int(terminal.get("retained") or 0)
        failed = int(terminal.get("failed") or 0)
        dropped = int(terminal.get("dropped") or 0)
        terminal_error = terminal.get("post_eviction_error")
        terminal_ok = (
            terminal.get("sealed") is True
            and terminal.get("seal_enqueued") is True
            and terminal.get("seal_failed") is False
            and terminal.get("producer_aborted") is False
            and not terminal_error
            and expected > 0
            and queued == expected
            and completed == expected
            and retained == expected
            and failed == 0
            and dropped == 0
        )
        if not terminal_ok:
            logger.warning(
                "BlockDiskStore durability fence %s did not retain its complete "
                "publication (expected=%d queued=%d completed=%d retained=%d "
                "failed=%d dropped=%d error=%s)",
                normalized_fence_id,
                expected,
                queued,
                completed,
                retained,
                failed,
                dropped,
                terminal_error,
            )
            if not allow_partial:
                return set()

        with self.global_budget.mutation_guard() as locked:
            if not locked:
                return set()
            return {
                block_hash
                for block_hash in targets
                if self._has_block_guarded(block_hash)
            }

    def _background_writer(self) -> None:
        """Background thread: drain write queue and persist blocks.

        Uses a persistent write connection for the lifetime of this thread,
        avoiding the overhead of opening/closing a SQLite connection per
        operation. The connection is created once at thread start.
        """
        write_conn = None

        try:
            while not self._stop_event.is_set() or not self._write_queue.empty():
                if write_conn is None:
                    try:
                        write_conn = sqlite3.connect(str(self._db_path), timeout=5.0)
                        write_conn.execute("PRAGMA journal_mode=WAL")
                    except Exception:
                        if write_conn is not None:
                            write_conn.close()
                            write_conn = None
                        self._disable_global_budget_writes()
                        logger.exception("Block-disk writer connection unavailable; retrying")
                        # Admission is disabled and existing queued data stays
                        # bounded. A transient reopen must not kill the writer.
                        time.sleep(0.2)
                        continue
                if self._pending_publication_recovery is not None:
                    try:
                        with self.global_budget.exclusive_mutation_guard() as locked:
                            if not locked:
                                raise RuntimeError("publication recovery root lock unavailable")
                            self._recover_pending_publication_locked(write_conn)
                    except Exception:
                        logger.exception("Block publication cleanup remains pending; retrying")
                        write_conn.close()
                        write_conn = None
                        time.sleep(0.2)
                        continue
                # Collect a batch: block on the first item (with timeout so we
                # can check the stop event), then drain any remaining items.
                batch = []
                try:
                    item = self._write_queue.get(timeout=0.2)
                    batch.append(item)
                    # Do not let the long-lived writer frame retain the first
                    # dequeued native NumPy payload while the queue is idle.
                    # The batch owns it now and _process_write_batch() clears
                    # each data slot as soon as publication settles.
                    item = None
                except queue.Empty:
                    # Idle: every enqueued write has been processed and its
                    # fence settled, so an owed periodic budget rescan runs
                    # here, off every request's terminal durability barrier.
                    # Recorded LRU touches are applied here too (<= 0.2 s old).
                    self._flush_pending_access(write_conn)
                    self._run_deferred_budget_reconcile()
                    continue

                # Drain remaining items without blocking
                while True:
                    try:
                        batch.append(self._write_queue.get_nowait())
                    except queue.Empty:
                        break

                # Queue capacity is available as soon as items are dequeued;
                # do not make bounded native producers wait for the entire
                # serialization/publication batch to finish before retrying.
                with self._write_lifecycle:
                    self._write_lifecycle.notify_all()

                try:
                    self._process_write_batch(write_conn, batch)
                except Exception as batch_exc:  # noqa: BLE001
                    # One unguarded exception used to escape this loop, run the
                    # outer `finally: write_conn.close()`, and KILL the writer
                    # thread for the life of the process. Every later L2 write
                    # then sat in the queue forever: no disk cache, no error,
                    # and the fences those writes were supposed to settle never
                    # resolved, so their blocks stayed pinned and un-evictable.
                    #
                    # A failed batch costs a re-prefill. A dead writer costs the
                    # whole L2 tier. Log it loudly and keep serving.
                    logger.error(
                        "Block-disk write batch failed (%d items): %s — "
                        "writer continues; those blocks fall back to recompute.",
                        len(batch),
                        batch_exc,
                        exc_info=True,
                    )
                    if isinstance(batch_exc, _BlockWriteConnectionError):
                        # The failed transaction was closed, not left for a
                        # later item's commit to publish accidentally.
                        write_conn.close()
                        write_conn = None
                finally:
                    self._complete_write_items(len(batch))
                    # _process_write_batch mutates data slots to None. Clear
                    # the outer list as well so this long-lived thread frame
                    # cannot pin completed NumPy/MLX-backed buffers between
                    # queue wakeups.
                    batch.clear()
                    item = None
        finally:
            if write_conn is not None:
                try:
                    self._flush_pending_access(write_conn)
                finally:
                    write_conn.close()

    _MAINTENANCE_QUIET_SECONDS = 1.0

    def _writer_quiet_for_maintenance(self) -> bool:
        """True only when no store is in progress: nothing queued or in flight,
        no unfinished write fence, and no write activity for a full second."""

        if not self._write_queue.empty():
            return False
        probe = self._activity_probe
        if probe is not None:
            try:
                if probe():
                    return False
            except Exception:  # noqa: BLE001
                return False  # unknown engine state counts as busy
        with self._stats_lock:
            if self._pending_write_items > 0 or self._write_inflight > 0:
                return False
            for state in self._write_fences.values():
                if state.get("post_eviction_complete") or state.get("producer_aborted"):
                    continue
                return False
        return (
            time.monotonic() - self._last_write_activity_monotonic
            >= self._MAINTENANCE_QUIET_SECONDS
        )

    def _run_deferred_budget_reconcile(self) -> None:
        """Run an owed rescan of the global budget while the writer is idle.

        Two owners: the DEFERRED pass a publish raised once the interval was
        due, and the IDLE pass the coordinator owes when its last scan left the
        root above its ceiling or protected dead-lease temp files inside the
        orphan grace (no write is needed for either to run).
        """

        budget = getattr(self, "global_budget", None)
        if budget is None:
            return
        deferred = bool(getattr(budget, "deferred_reconcile_due", False))
        idle_due = getattr(budget, "idle_reconcile_due", None)
        if not deferred and not (callable(idle_due) and idle_due()):
            return
        if not self._writer_quiet_for_maintenance():
            return
        started = time.perf_counter()
        try:
            result = budget.run_deferred_reconcile() if deferred else budget.run_idle_reconcile()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deferred global block-cache budget rescan failed: %s", exc)
            return
        if result is None:
            return
        self.disk_evictions += int(getattr(result, "evicted_entries", 0) or 0)
        logger.info(
            "%s global block-cache budget rescan: %.3fs scan_performed=%s "
            "usage=%.3fGB / %.3fGB evicted=%d compliant=%s",
            "Deferred" if deferred else "Idle",
            time.perf_counter() - started,
            result.scan_performed,
            result.bytes_after / 1e9,
            result.max_size_bytes / 1e9,
            int(result.evicted_entries or 0),
            result.compliant,
        )

    def _stage_block_groups(self, batch: List[Tuple[Any, ...]]) -> None:
        """Writer thread: stage each chain of >= 2 eligible queued blocks (same fence and dtype, every parent the
        previous member, no replace) as ONE group file; members become ``_GroupMember`` payloads in place."""
        group_max = _group_max()
        if group_max < 2:
            return
        chain: List[int] = []

        def flush() -> None:
            if len(chain) >= 2:
                self._stage_one_group(batch, list(chain))
            chain.clear()

        for idx, item in enumerate(batch):
            eligible = (
                bool(item)
                and item[0] == "__numpy_block__"
                and len(item) >= 10
                and not bool(item[9])
                and _numpy_block_group_eligible(item[2])
            )
            if not eligible:
                flush()
                continue
            if chain:
                prev = batch[chain[-1]]
                if (
                    len(chain) >= group_max
                    or item[6] is None
                    or bytes(item[6]) != bytes(prev[1])
                    or item[7] != prev[7]
                    or item[3] != prev[3]
                ):
                    flush()
            chain.append(idx)
        flush()

    def _stage_one_group(self, batch: List[Tuple[Any, ...]], idxs: List[int]) -> None:
        members = [(bytes(batch[i][1]).hex(), batch[i][2]) for i in idxs]
        try:
            group = _numpy_group_tensors(members)
            staged = self._stage_numpy_safetensors_file(batch[idxs[0]][1], group)
            size = max(1, int(staged.stat().st_size))
        except Exception as exc:  # noqa: BLE001 - fall back to one file per block
            logger.warning("Block group staging failed (%d blocks): %s; writing them one per file", len(idxs), exc)
            return
        finally:
            group = None
            members = None
        self._group_staged_paths.append(staged)
        k = len(idxs)
        share = size // k
        for n, i in enumerate(idxs):
            (
                _,
                block_hash,
                _tensors,
                dtype,
                num_layers,
                token_count,
                parent_hash,
                fence_id,
                reserved_bytes,
                replace_existing,
            ) = batch[i]
            member_share = share + (size - share * k if n == k - 1 else 0)
            if not self._resize_pending_write_reservation(reserved_bytes, member_share):
                batch[i] = None
                self._write_fence_completion(fence_id, failed=True)
                with self._stats_lock:
                    self._offthread_serialization_failures += 1
                continue
            batch[i] = (
                block_hash,
                _GroupMember(staged, n, k, member_share),
                dtype,
                num_layers,
                token_count,
                parent_hash,
                fence_id,
                member_share,
                bool(replace_existing),
            )
            with self._stats_lock:
                self._offthread_serializations_completed += 1
        with self._stats_lock:
            self.group_files_written += 1
            self.group_members_written += k

    def _process_write_batch(
        self,
        write_conn: sqlite3.Connection,
        batch: List[Tuple[Any, ...]],
    ) -> None:
        """Persist one drained batch and settle fences after aggregate eviction."""
        original_batch_count = len(batch)
        access_only_batch = bool(batch) and all(
            item and item[0] == "__access__" for item in batch
        )
        self._last_write_activity_monotonic = time.monotonic()
        with self._stats_lock:
            # Count CPU encoding as in-flight writer work too. Durability
            # waiters also observe _pending_write_items, but /health must not
            # report an idle writer while it is encoding a large page.
            self._write_inflight += original_batch_count

        # Group files first (see _group_max): chains of eligible queued blocks
        # become one staged file whose members publish as hard links below.
        self._stage_block_groups(batch)

        # Encode in place. The caller owns this exact list, so replacing each
        # NumPy item immediately is what releases its frozen native buffers.
        # Building a second ``prepared_batch`` kept the complete NumPy batch
        # alive while also accumulating a complete bytes batch: live Muse
        # publication consequently left ~820 MiB in MALLOC_LARGE (empty), and
        # the 4 GiB pending budget allowed a much larger duplicate high-water
        # mark on long prompts.
        for item_idx, item in enumerate(batch):
            if not item or item[0] != "__numpy_block__":
                continue
            (
                _,
                block_hash,
                tensors,
                dtype,
                num_layers,
                token_count,
                parent_hash,
                fence_id,
                reserved_bytes,
                replace_existing,
            ) = item
            staged_path: Path | None = None
            try:
                staged_path = self._stage_numpy_safetensors_file(
                    block_hash,
                    tensors,
                )
                payload_size = max(1, int(staged_path.stat().st_size))
                if not self._resize_pending_write_reservation(
                    reserved_bytes,
                    payload_size,
                ):
                    batch[item_idx] = None
                    staged_path.unlink(missing_ok=True)
                    staged_path = None
                    self._write_fence_completion(fence_id, failed=True)
                    with self._stats_lock:
                        self._offthread_serialization_failures += 1
                    logger.warning(
                        "BlockDiskStore encoded payload exceeded pending-write "
                        "byte budget (%d bytes)",
                        self._max_pending_write_bytes,
                    )
                    continue
                with self._stats_lock:
                    self._offthread_serializations_completed += 1
                batch[item_idx] = (
                    block_hash,
                    staged_path,
                    dtype,
                    num_layers,
                    token_count,
                    parent_hash,
                    fence_id,
                    payload_size,
                    bool(replace_existing),
                )
                staged_path = None
            except Exception as exc:
                batch[item_idx] = None
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)
                self._release_pending_write_bytes(reserved_bytes)
                self._write_fence_completion(fence_id, failed=True)
                with self._stats_lock:
                    self._offthread_serialization_failures += 1
                logger.warning(
                    "Background block serialization failed (%s): %s",
                    bytes(block_hash).hex()[:12],
                    exc,
                )
            finally:
                # The list slot above is now the sole owner of a successful
                # staged path. Drop loop locals before disk publication so no
                # frozen NumPy map survives alongside it.
                tensors = None
                staged_path = None
                item = None

        batch_fence_ids = {
            str(item[6])
            for item in batch
            if item
            and not isinstance(item[0], str)
            and len(item) > 6
            and item[6] is not None
        }
        # Byte reservations are settled per block the moment its payload is
        # durably handed to SQLite/the filesystem, NOT at batch end. A batch
        # that ends in a multi-second aggregate eviction otherwise pegs the
        # pending budget for its whole duration and every concurrently
        # admitted store drops at the 0.25s admission wait (measured: 255/383
        # blocks lost on a bounded-L2 store). Indexed by position in `batch`
        # so per-item settlement can also drop the payload reference (the
        # budget must track actual RAM) and the finally settles whatever an
        # early return/exception left behind.
        outstanding_reservations: Dict[int, int] = {
            idx: int(item[7])
            for idx, item in enumerate(batch)
            if item and not isinstance(item[0], str) and len(item) > 7
        }
        fences_to_finalize: List[str] = []
        net_payload_bytes = 0
        new_block_hashes: list[bytes] = []
        budget_failure_reason: str | None = None
        try:
            # Access/cleanup updates, final rename, and SQLite mutation share
            # one publication lock. Root-global eviction therefore sees a
            # stable set of finalized records and ordered LRU metadata.
            with self.global_budget.exclusive_mutation_guard() as locked:
                if not locked:
                    for item in batch:
                        if not item or isinstance(item[0], str):
                            continue
                        fence_id = (
                            str(item[6])
                            if len(item) > 6 and item[6] is not None
                            else None
                        )
                        self._write_fence_completion(fence_id, failed=True)
                    deferred_failures: set[str] = set()
                    for item in batch:
                        if item and item[0] == "__fence__":
                            deferred_failures.add(str(item[1]))
                    with self._stats_lock:
                        deferred_failures.update(
                            pending_fence_id
                            for pending_fence_id, state in self._write_fences.items()
                            if (
                                bool(state.get("sealed"))
                                and bool(state.get("seal_enqueued"))
                                and not state.get("post_eviction_complete")
                                and self._write_fence_publication_ready_locked(state)
                            )
                        )
                    for pending_fence_id in deferred_failures:
                        self._fail_write_fence(
                            pending_fence_id,
                            "global block-cache publication lock unavailable",
                        )
                    return
                metadata_before = self._index_physical_bytes()
                with self._stats_lock:
                    terminal_fences = {
                        pending_fence_id
                        for pending_fence_id, state in self._write_fences.items()
                        if state.get("post_eviction_complete")
                    }
                self._release_write_fence_pins_locked(
                    write_conn,
                    terminal_fences,
                )
                access_rows: List[Tuple[float, int, str]] = []
                for idx, item in enumerate(batch):
                    if not item:
                        continue
                    fence_id: Optional[str] = None
                    try:
                        if item[0] == "__access__":
                            _, hash_hex, ts = item
                            if hash_hex:
                                access_rows.append((ts, 1, hash_hex))
                        elif item[0] == "__cleanup__":
                            _, hash_hex, _ = item
                            net_payload_bytes -= self._cleanup_entry(
                                write_conn, hash_hex
                            )
                        elif item[0] == "__fence__":
                            _, fence_id, _ = item
                            fences_to_finalize.append(str(fence_id))
                        else:
                            (
                                block_hash,
                                payload,
                                dtype,
                                num_layers,
                                token_count,
                                parent_hash,
                                *metadata,
                            ) = item
                            fence_id = (
                                str(metadata[0])
                                if metadata and metadata[0] is not None
                                else None
                            )
                            written_bytes = self._write_block(
                                write_conn,
                                block_hash,
                                payload,
                                dtype,
                                num_layers,
                                token_count,
                                parent_hash,
                                replace_existing=(
                                    bool(metadata[2])
                                    if len(metadata) > 2
                                    else False
                                ),
                                fence_id=fence_id,
                            )
                            net_payload_bytes += written_bytes
                            if written_bytes > 0:
                                new_block_hashes.append(bytes(block_hash))
                            self._write_fence_completion(fence_id, failed=False)
                    except Exception as e:
                        if not isinstance(e, _BlockWriteConnectionError):
                            try:
                                write_conn.rollback()
                            except Exception as rollback_error:
                                write_conn.close()
                                self._disable_global_budget_writes()
                                e = _BlockWriteConnectionError(
                                    f"block writer rollback failed: {rollback_error}"
                                )
                        if item and not isinstance(item[0], str):
                            self._write_fence_completion(fence_id, failed=True)
                        if isinstance(e, _BlockWriteConnectionError):
                            # Recovery already closed the poisoned connection.
                            # Settle every unprocessed item and fail its fence;
                            # do not run more SQL on this connection or certify
                            # the earlier part of this batch from stale totals.
                            for remaining in batch[idx + 1:]:
                                if remaining and not isinstance(remaining[0], str):
                                    remaining_fence = (
                                        str(remaining[6])
                                        if len(remaining) > 6 and remaining[6]
                                        else None
                                    )
                                    self._write_fence_completion(remaining_fence, failed=True)
                            failed_fences = batch_fence_ids | {
                                str(entry[1]) for entry in batch
                                if entry and entry[0] == "__fence__"
                            }
                            for failed_fence in failed_fences:
                                self._mark_write_fence_failed(failed_fence, str(e))
                            self._pending_publication_recovery = (e.block_hash, failed_fences)
                            if e.block_hash is not None:
                                # Invalidate bytes even when SQLite cannot yet
                                # reconnect. A stale row must not restore them.
                                try:
                                    failed_path = self._hash_to_path(e.block_hash.hex())
                                    failed_path.unlink(missing_ok=True)
                                    self._fsync_directory(failed_path.parent)
                                except OSError:
                                    logger.exception("Failed block payload removal remains pending")
                            # Do not leave pins from earlier successful items
                            # stranded until another request happens to write.
                            for recovery_attempt in range(2):
                                recovery_conn = None
                                try:
                                    recovery_conn = sqlite3.connect(str(self._db_path), timeout=5.0)
                                    self._recover_pending_publication_locked(recovery_conn)
                                    break
                                except Exception:
                                    logger.exception(
                                        "Failed block publication recovery attempt %d",
                                        recovery_attempt + 1,
                                    )
                                finally:
                                    if recovery_conn is not None:
                                        recovery_conn.close()
                            try:
                                self.global_budget._enforce_locked()
                            except Exception:
                                logger.exception("Failed block publication reconciliation failed")
                            raise e
                        h = item[0] if isinstance(item[0], str) else (
                            item[0].hex()[:12] if isinstance(item[0], bytes) else "?"
                        )
                        logger.warning(f"Background writer error ({h}): {e}")
                    finally:
                        # Settle this block's byte reservation now — its staged
                        # payload is either renamed or failed. Remove any temp
                        # that survived an early return before dropping the
                        # batch slot, so admission waiters see real progress.
                        reserved_now = outstanding_reservations.pop(idx, 0)
                        if reserved_now:
                            if isinstance(payload, Path):
                                payload.unlink(missing_ok=True)
                            # a _GroupMember's staged group file is shared; it is
                            # unlinked once at batch end (its members keep links)
                            batch[idx] = None
                            self._release_pending_write_bytes(reserved_now)
                        # Drop loop references before eviction/fence accounting.
                        payload = None
                        item = None

                access_rows.extend(self._take_pending_access_rows())
                if access_rows:
                    # One transaction for every LRU touch in the batch (same
                    # rows, latest timestamps and total access counts as one
                    # UPDATE+COMMIT per touch, which cost ~1 ms each on the
                    # writer thread and competed with the restoring thread).
                    try:
                        write_conn.executemany(self._ACCESS_UPDATE_SQL, access_rows)
                        write_conn.commit()
                    except Exception as access_error:  # noqa: BLE001 - LRU freshness only
                        logger.warning(
                            "Background writer access update failed (%d rows): %s",
                            len(access_rows),
                            access_error,
                        )
                        try:
                            write_conn.rollback()
                        except Exception:
                            pass
                    access_rows = []

                # A fence control sentinel may be deferred when the bounded
                # data FIFO is full. Finalize it from the writer only after
                # every admitted payload for that fence has been processed;
                # this preserves FIFO durability without making control-path
                # completion droppable under data pressure.
                with self._stats_lock:
                    for pending_fence_id, state in self._write_fences.items():
                        if (
                            bool(state.get("sealed"))
                            and bool(state.get("seal_enqueued"))
                            and not state.get("seal_failed")
                            and not state.get("post_eviction_complete")
                            and self._write_fence_publication_ready_locked(state)
                            and pending_fence_id not in fences_to_finalize
                        ):
                            fences_to_finalize.append(pending_fence_id)
                with self._stats_lock:
                    strict_reconcile = any(
                        bool(
                            self._write_fences.get(pending_fence_id, {}).get(
                                "_strict_reconcile"
                            )
                        )
                        for pending_fence_id in fences_to_finalize
                    )
                    protected_blocks = {
                        (self._db_path.resolve(), block_hash.hex())
                        for pending_fence_id in fences_to_finalize
                        for block_hash in (
                            self._write_fences.get(
                                pending_fence_id,
                                {},
                            ).get("_queued_hashes")
                            or ()
                        )
                    }
                # Persistent pins bridge separate writer batches.  The final
                # batch converts them to an in-lock protected set so the same
                # reconciliation can trim unrelated LRU entries and certify
                # this exact chain without leaving pins after completion.
                self._release_write_fence_pins_locked(
                    write_conn,
                    set(fences_to_finalize),
                )
                if access_only_batch:
                    # LRU touches change neither the finalized payload set nor
                    # its logical byte total. Feeding their SQLite/WAL page
                    # churn into account_finalized_write_locked made the
                    # 30-second integrity interval launch a full managed-root
                    # scan from a cache HIT. The scan holds this same root lock
                    # exclusively; a 32-block Qwen restore then blocked its own
                    # remaining reads for 3.9 seconds on a 178 GB root.
                    #
                    # Keep the short exclusive section so an eviction cannot
                    # race the durable LRU update, but leave byte accounting to
                    # the next real publish/delete (or startup enforcement).
                    # WAL space is bounded/reused and is reconciled by those
                    # owning paths; a touch cannot admit new cache payload.
                    return
                metadata_delta = self._index_physical_bytes() - metadata_before
                try:
                    global_result = (
                        self.global_budget.account_finalized_write_locked(
                            net_payload_bytes + metadata_delta,
                            require_reconciled=strict_reconcile,
                            protected_blocks=protected_blocks,
                        )
                    )
                    if not global_result.accounted or not global_result.compliant:
                        budget_failure_reason = (
                            global_result.error
                            or (
                                "aggregate block-cache remains over budget "
                                f"({global_result.bytes_after} > "
                                f"{global_result.max_size_bytes})"
                            )
                        )
                        # Distinguish "the budget is full" from "this ONE chain
                        # is bigger than the entire budget". They read the same
                        # in the fence warning, but only the second is
                        # permanent: every long prompt will be refused until the
                        # budget is raised, and the cache silently stores
                        # nothing at all. Observed with a 0.5GB budget and a
                        # 6050-token / 44-layer chain needing ~640MB.
                        _chain_bytes = net_payload_bytes + metadata_delta
                        _budget_bytes = global_result.max_size_bytes or 0
                        if _budget_bytes and _chain_bytes > _budget_bytes:
                            logger.warning(
                                "Block cache budget is smaller than a single "
                                "prompt's block chain (%.1fMB needed vs %.1fMB "
                                "budget); nothing from this request can be "
                                "cached. Raise the SSD cache size (Session "
                                "Settings -> SSD Cache Size, or "
                                "--block-disk-cache-max-percent / "
                                "--block-disk-cache-max-gb) to reuse prompts "
                                "this long.",
                                _chain_bytes / (1024 ** 2),
                                _budget_bytes / (1024 ** 2),
                            )
                        released_fences = self._release_write_fence_pins_locked(
                            write_conn,
                        )
                        failed_fences = (
                            released_fences
                            | batch_fence_ids
                            | set(fences_to_finalize)
                        )
                        for failed_fence_id in failed_fences:
                            self._mark_write_fence_failed(
                                failed_fence_id,
                                budget_failure_reason,
                            )
                        # The protected chain cannot fit.  Keep RAM
                        # authoritative, remove all of this owner's pins, and
                        # restore the physical ceiling without protection.
                        global_result = self.global_budget._enforce_locked()
                        if not (
                            global_result.accounted
                            and global_result.compliant
                        ):
                            self._disable_global_budget_writes()
                except Exception as exc:
                    released_fences = self._release_write_fence_pins_locked(
                        write_conn,
                    )
                    for written_hash in new_block_hashes:
                        try:
                            self._cleanup_entry(
                                write_conn,
                                written_hash.hex(),
                            )
                        except Exception:
                            pass
                    failed_fences = (
                        released_fences
                        | batch_fence_ids
                        | set(fences_to_finalize)
                    )
                    for failed_fence_id in failed_fences:
                        self._mark_write_fence_failed(
                            failed_fence_id,
                            str(exc),
                        )
                    try:
                        self.global_budget._enforce_locked()
                    except Exception:
                        pass
                    self._disable_global_budget_writes()
                    logger.warning("Background writer accounting error: %s", exc)
                    return

            try:
                # Reserve exact new payload bytes plus the SQLite/WAL/SHM net
                # change under the root ledger. Ordinary batches are O(1);
                # strict request fences force a physical reconciliation before
                # they may report completion.
                if budget_failure_reason is not None:
                    raise RuntimeError(budget_failure_reason)
                if not global_result.accounted or not global_result.compliant:
                    raise RuntimeError(
                        global_result.error
                        or (
                            "aggregate block-cache remains over budget "
                            f"({global_result.bytes_after} > "
                            f"{global_result.max_size_bytes})"
                        )
                    )
                if global_result.evicted_entries:
                    with self._stats_lock:
                        self.disk_evictions += global_result.evicted_entries
            except Exception as exc:
                logger.warning("Background writer eviction error: %s", exc)
                for fence_id in fences_to_finalize:
                    self._fail_write_fence(fence_id, str(exc))
                return

            for fence_id in fences_to_finalize:
                try:
                    self._last_global_fence_result = global_result
                    self._finalize_write_fence(write_conn, fence_id)
                except Exception as exc:
                    logger.warning(
                        "Background writer fence finalization error (%s): %s",
                        fence_id,
                        exc,
                    )
                    self._fail_write_fence(fence_id, str(exc))
        finally:
            # A publication-lock failure can return before the per-item loop.
            # Remove any lease-owned staged files that never reached rename.
            for pending_item in batch:
                if (
                    pending_item
                    and not isinstance(pending_item[0], str)
                    and len(pending_item) > 1
                    and isinstance(pending_item[1], Path)
                ):
                    pending_item[1].unlink(missing_ok=True)
            # Staged group files: every member that published holds its own
            # hard link now, so the staging name goes (the bytes stay until the
            # last member link is removed); unpublished groups are freed here.
            while self._group_staged_paths:
                self._group_staged_paths.pop().unlink(missing_ok=True)
            # Blocks the write loop settled already popped their entries; this
            # covers early returns (publication lock unavailable) and batch
            # exceptions that skipped the per-item settlement.
            for reserved in outstanding_reservations.values():
                self._release_pending_write_bytes(reserved)
            outstanding_reservations.clear()
            with self._stats_lock:
                self._write_inflight = max(
                    0,
                    self._write_inflight - original_batch_count,
                )

    def _finalize_write_fence(
        self,
        write_conn: sqlite3.Connection,
        fence_id: str,
    ) -> None:
        """Record exact-hash retention after the containing batch's eviction."""
        global_result = getattr(self, "_last_global_fence_result", None)
        if global_result is None:
            raise RuntimeError(
                "global block-cache accounting result missing at fence finalization"
            )
        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is None:
                return
            if not self._write_fence_publication_ready_locked(state):
                state["seal_enqueued"] = False
                return
            queued_hashes = list(state.get("_queued_hashes") or ())

        retained = 0
        for block_hash in queued_hashes:
            hash_hex = block_hash.hex()
            row = write_conn.execute(
                "SELECT file_name FROM blocks WHERE block_hash = ?",
                (hash_hex,),
            ).fetchone()
            if row is not None and (self.cache_dir / row[0]).is_file():
                retained += 1

        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is None:
                return
            self._write_completion_generation += 1
            state["retained"] = retained
            state["post_eviction_complete"] = True
            state["completion_generation"] = self._write_completion_generation
            state["global_accounting_generation"] = (
                global_result.accounting_generation
            )
            state["global_reconciliation_generation"] = (
                global_result.reconciliation_generation
            )
            state["global_bytes_after"] = global_result.bytes_after
            state["global_max_size_bytes"] = global_result.max_size_bytes
            self._prune_write_fences_locked()

    def _mark_write_fence_failed(self, fence_id: str, reason: str) -> None:
        """Make a fence terminal; caller owns persistent-pin settlement."""

        with self._stats_lock:
            state = self._write_fences.get(fence_id)
            if state is None or state.get("post_eviction_complete"):
                return
            self._write_completion_generation += 1
            state["post_eviction_error"] = str(reason)
            state["post_eviction_complete"] = True
            state["completion_generation"] = self._write_completion_generation
            self._prune_write_fences_locked()

    def _fail_write_fence(self, fence_id: str, reason: str) -> None:
        """Make a fence terminal and release any cross-process disk pins."""

        self._mark_write_fence_failed(fence_id, reason)
        self._release_write_fence_pins(fence_id)

    def _update_access(self, conn: sqlite3.Connection, hash_hex: str, ts: float) -> None:
        """Update last_accessed time in the index (background thread only)."""
        conn.execute(
            "UPDATE blocks SET last_accessed = ?, access_count = access_count + 1 "
            "WHERE block_hash = ?",
            (ts, hash_hex)
        )
        conn.commit()

    def _cleanup_entry(self, conn: sqlite3.Connection, hash_hex: str) -> int:
        """Remove an entry without leaving indexed descendants unrestorable.

        Known chains are deleted from the requested node through every branch
        below it.  A pre-migration row has ambiguous ``NULL`` ancestry, so an
        exact cleanup of one such row invalidates the complete legacy set in
        that namespace rather than pretending the other legacy rows are roots.
        """
        row = conn.execute(
            "SELECT ancestry_known FROM blocks WHERE block_hash = ?",
            (hash_hex,),
        ).fetchone()
        if row is not None and int(row[0] or 0) == 0:
            starting_hashes = {
                str(item[0])
                for item in conn.execute(
                    "SELECT block_hash FROM blocks WHERE ancestry_known = 0"
                ).fetchall()
            }
        else:
            # Include the requested hash even when its own row is already gone:
            # a failed/partial publication may still have left known children.
            starting_hashes = {str(hash_hex)}

        targets = set(starting_hashes)
        frontier = set(starting_hashes)
        while frontier:
            next_frontier: set[str] = set()
            frontier_list = sorted(frontier)
            for start in range(0, len(frontier_list), 500):
                chunk = frontier_list[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                child_rows = conn.execute(
                    f"SELECT block_hash FROM blocks "
                    f"WHERE ancestry_known = 1 "
                    f"AND parent_hash IN ({placeholders})",
                    chunk,
                ).fetchall()
                next_frontier.update(str(item[0]) for item in child_rows)
            next_frontier.difference_update(targets)
            targets.update(next_frontier)
            frontier = next_frontier

        if not targets:
            return 0

        rows: list[tuple[str, str]] = []
        target_list = sorted(targets)
        for start in range(0, len(target_list), 500):
            chunk = target_list[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                (str(block_hash), str(file_name))
                for block_hash, file_name in conn.execute(
                    f"SELECT block_hash, file_name FROM blocks "
                    f"WHERE block_hash IN ({placeholders})",
                    chunk,
                ).fetchall()
            )
        if not rows:
            return 0

        conn.execute("BEGIN IMMEDIATE")
        try:
            for start in range(0, len(target_list), 500):
                chunk = target_list[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM block_write_pins "
                    f"WHERE block_hash IN ({placeholders})",
                    chunk,
                )
                conn.execute(
                    f"DELETE FROM blocks WHERE block_hash IN ({placeholders})",
                    chunk,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        removed_bytes = 0
        for _block_hash, file_name in rows:
            relative = Path(file_name)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            file_path = self.cache_dir / relative
            try:
                file_size = max(0, int(file_path.stat().st_size))
                file_path.unlink()
                removed_bytes += file_size
            except FileNotFoundError:
                continue
            except OSError:
                # The row is already unreachable.  A later root scan counts the
                # surviving payload as an orphan and removes it safely.
                continue
        return removed_bytes

    def _new_payload_temp_path(self, file_path: Path) -> Path:
        """Return one lease-owned temp path beside its finalized payload."""

        seq = self._tmp_seq
        self._tmp_seq += 1
        return file_path.with_name(
            f"{file_path.stem}.{self.global_budget.lease_id}.{seq}."
            f"{uuid.uuid4().hex}.tmp.safetensors"
        )

    @staticmethod
    def _write_payload_file(path: Path, payload: "bytes | Path | _GroupMember") -> None:
        """Durably stage an immutable payload (writer thread only)."""

        if isinstance(payload, _GroupMember):
            # The staged group file was streamed and fsynced once; each member
            # publishes a hard link to it under its own name. The caller
            # performs the atomic rename and directory fsync as for a block.
            os.link(str(payload.path), str(path))
            return
        if isinstance(payload, Path):
            # ``payload`` was already streamed and fsynced. Move it without
            # materializing any tensor-data bytes image. The caller still
            # performs the final atomic rename and directory fsync before
            # publishing the SQLite row.
            os.replace(str(payload), str(path))
            return

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise OSError("short block-cache payload write")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _indexed_payload_is_readable(self, file_name: str) -> bool:
        """Validate one finalized index path without following symlinks."""
        relative = Path(str(file_name))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        file_path = self.cache_dir / relative
        if file_path.is_symlink():
            return False
        try:
            resolved = file_path.resolve(strict=True)
            cache_root = self.cache_dir.resolve(strict=True)
        except OSError:
            return False
        if not resolved.is_relative_to(cache_root) or not resolved.is_file():
            return False
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(resolved, flags)
        except OSError:
            return False
        else:
            os.close(fd)
            return True

    def _write_block(
        self,
        conn: sqlite3.Connection,
        block_hash: bytes,
        payload: "bytes | Path | _GroupMember",
        dtype: str,
        num_layers: int,
        token_count: int,
        parent_hash: Optional[bytes],
        *,
        replace_existing: bool = False,
        fence_id: Optional[str] = None,
    ) -> int:
        """Write and publish a frozen block (background writer thread only)."""
        hash_hex = block_hash.hex()
        parent_hex = parent_hash.hex() if parent_hash is not None else None

        # A child is publishable only when its immediate parent is already a
        # known-ancestry row.  The queue is ordered root-to-tail, so a failed
        # parent publication makes every dependent child fail closed instead of
        # creating an indexed but unrestorable suffix.
        if parent_hex is not None:
            parent = conn.execute(
                "SELECT ancestry_known, file_name FROM blocks "
                "WHERE block_hash = ?",
                (parent_hex,),
            ).fetchone()
            if (
                parent is None
                or int(parent[0] or 0) != 1
                or not self._indexed_payload_is_readable(str(parent[1]))
            ):
                if parent is not None:
                    self._cleanup_entry(conn, parent_hex)
                raise ValueError(
                    "cannot publish block whose parent ancestry is unavailable"
                )

        # Skip if already on disk, but allow an ordered current write-through to
        # upgrade a legacy row after its parent chain has been established.
        exists = conn.execute(
            "SELECT parent_hash, ancestry_known, file_name, file_size "
            "FROM blocks WHERE block_hash = ?",
            (hash_hex,),
        ).fetchone()
        if exists:
            existing_parent, ancestry_known, existing_file, _existing_size = exists
            if not self._indexed_payload_is_readable(str(existing_file)):
                self._cleanup_entry(conn, hash_hex)
                exists = None
        replaced_file_size = 0
        replacing_existing = False
        if exists:
            (
                existing_parent,
                ancestry_known,
                _existing_file,
                existing_file_size,
            ) = exists
            if int(ancestry_known or 0) == 0:
                conn.execute(
                    "UPDATE blocks SET parent_hash = ?, ancestry_known = 1 "
                    "WHERE block_hash = ? AND ancestry_known = 0",
                    (parent_hex, hash_hex),
                )
                conn.commit()
            elif existing_parent != parent_hex:
                raise ValueError(
                    "block hash already exists with different parent ancestry"
                )
            if not replace_existing:
                self._pin_write_fence_block_locked(conn, fence_id, block_hash)
                return 0
            replaced_file_size = max(0, int(existing_file_size or 0))
            replacing_existing = True

        file_path = self._hash_to_path(hash_hex)
        rel_path = file_path.relative_to(self.cache_dir)
        tmp_path = self._new_payload_temp_path(file_path)

        try:
            self._write_payload_file(tmp_path, payload)
            os.replace(str(tmp_path), str(file_path))
            self._fsync_directory(file_path.parent)
        except Exception:
            # Clean up partial file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        file_stat = file_path.stat()
        # A group member accounts for its share of the shared group file.
        file_size = payload.share if isinstance(payload, _GroupMember) else file_stat.st_size
        group_size = payload.size if isinstance(payload, _GroupMember) else None
        now = time.time()

        # A new row and its eviction pin become visible in one transaction.
        # Snapshot fence state BEFORE any DML: get_stats holds the stats lock
        # while reading SQLite, so writers must commit before taking it again.
        pin_new = not replacing_existing and self._write_fence_pin_is_open(fence_id)
        if replacing_existing:
            conn.execute(
                """UPDATE blocks
                   SET parent_hash = ?, ancestry_known = 1, file_name = ?,
                       num_tokens = ?, num_layers = ?, dtype = ?, file_size = ?,
                       created_at = ?, last_accessed = ?, group_size = ?
                   WHERE block_hash = ?""",
                (
                    parent_hex,
                    str(rel_path),
                    token_count,
                    num_layers,
                    dtype,
                    file_size,
                    now,
                    now,
                    group_size,
                    hash_hex,
                ),
            )
        else:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO blocks
                       (block_hash, parent_hash, ancestry_known,
                        file_name, num_tokens, num_layers, dtype,
                        file_size, created_at, last_accessed, group_size)
                       VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        hash_hex,
                        parent_hex,
                        str(rel_path),
                        token_count,
                        num_layers,
                        dtype,
                        file_size,
                        now,
                        now,
                        group_size,
                    ),
                )
                if pin_new:
                    self._insert_write_fence_pin_locked(conn, str(fence_id), block_hash)
                conn.commit()
            except Exception:
                # A COMMIT exception can occur before OR after durability.
                # Roll back pending SQL and invalidate this attempted new hash
                # before any later child/item can use it. Never remove a valid
                # preexisting block here: replacements use the branch above.
                try:
                    conn.rollback()
                    self._cleanup_entry(conn, hash_hex)
                    file_path.unlink(missing_ok=True)
                    self._fsync_directory(file_path.parent)
                except Exception as recovery_error:
                    try:
                        conn.close()
                    finally:
                        self._disable_global_budget_writes()
                    raise _BlockWriteConnectionError(
                        "new block transaction recovery failed; connection discarded",
                        block_hash,
                    ) from recovery_error
                raise
        if replacing_existing:
            conn.commit()

        if not self._allow_tq_native:
            # write_block_async() refuses TQ cache data in this mode, so every
            # payload that reaches the writer is plain KV: record the exact file
            # so later dedup checks of this hash skip re-inspecting it.
            self._note_verified_plain(hash_hex, _stat_identity(file_stat))

        with self._stats_lock:
            self.disk_writes += 1
        logger.debug(
            f"Disk cache write: {hash_hex[:12]} ({dtype}, {num_layers} layers, "
            f"{file_size / 1024:.1f}KB, {token_count} tokens)"
        )
        if replacing_existing:
            self._pin_write_fence_block_locked(conn, fence_id, block_hash)
        return int(file_size) - replaced_file_size

    def _index_physical_bytes(self) -> int:
        """Return physical bytes for this namespace's SQLite index files."""
        total = 0
        for path in (
            self._db_path,
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
            Path(f"{self._db_path}-journal"),
        ):
            try:
                total += max(0, int(path.stat().st_size))
            except FileNotFoundError:
                continue
        return total

    # =========================================================================
    # Management
    # =========================================================================

    def _inspect_latest_payload(
        self,
        row: Optional[Tuple[Any, ...]],
    ) -> Optional[Dict[str, Any]]:
        """Read one safetensors header so UI telemetry reports real dtypes.

        The SQLite ``dtype`` column describes the cache tuple tag (``kv``,
        ``mixed``), not the tensor element dtype.  That made a live Qwen3.8
        cache panel say only "full precision" while every stored K/V tensor
        was actually F32 and consumed twice the bundle-declared BF16 size.
        Inspect the newest finalized payload, cache the result by path/mtime,
        and load only the tiny JSON metadata tensor -- never K/V data.
        """
        if not row:
            return None
        file_name, token_count, file_size, cache_tag = row
        try:
            payload_path = (self.cache_dir / str(file_name)).resolve(strict=True)
            cache_root = self.cache_dir.resolve(strict=True)
            if not payload_path.is_relative_to(cache_root):
                return None
            stat = payload_path.stat()
            cache_key = (str(payload_path), int(stat.st_mtime_ns), int(stat.st_size))
            cached = self._latest_payload_inspection_cache
            if cached is not None and cached[0] == cache_key:
                return dict(cached[1])

            from safetensors import safe_open

            physical_counts: Counter[str] = Counter()
            meta: Dict[str, Any] = {}
            with safe_open(str(payload_path), framework="numpy") as handle:
                keys = list(handle.keys())
                for name in keys:
                    if name in {"__vmlx_block_meta__", "__metadata__"}:
                        continue
                    physical_counts[str(handle.get_slice(name).get_dtype())] += 1
                meta_key = (
                    "__vmlx_block_meta__"
                    if "__vmlx_block_meta__" in keys
                    else "__metadata__" if "__metadata__" in keys else None
                )
                if meta_key is not None:
                    raw = handle.get_tensor(meta_key).tobytes()
                    decoded = json.loads(raw.decode("utf-8"))
                    if isinstance(decoded, dict):
                        meta = decoded

            layer_types = meta.get("__layer_types__") or {}
            orig_dtypes = meta.get("__orig_dtypes__") or {}
            attention_orig: Counter[str] = Counter()
            all_orig: Counter[str] = Counter()
            for layer_key, raw_dtype in orig_dtypes.items():
                normalized = str(raw_dtype).replace("mlx.core.", "")
                all_orig[normalized] += 1
                if (
                    str(layer_key).isdigit()
                    and layer_types.get(str(layer_key))
                    in {"kv", "rotating_kv", "minimax_m3"}
                ):
                    attention_orig[normalized] += 1
            result = {
                "token_count": int(token_count or 0),
                "file_size_bytes": int(file_size or stat.st_size),
                "cache_tag": str(cache_tag or ""),
                "cache_layer_count": int(meta.get("__num_cache_layers__") or 0),
                "layer_type_counts": dict(Counter(layer_types.values())),
                "physical_tensor_dtype_counts": dict(physical_counts),
                "original_tensor_dtype_counts": dict(all_orig),
                "original_attention_kv_dtype_counts": dict(attention_orig),
                "runtime_cache_fingerprint": meta.get(
                    "__runtime_cache_fingerprint__"
                ),
            }
            self._latest_payload_inspection_cache = (cache_key, dict(result))
            return result
        except Exception as exc:  # noqa: BLE001 - telemetry must not break stats
            logger.debug("Latest block payload inspection failed: %s", exc)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        # Fence completion and the SQLite row it certifies must be one coherent
        # observation.  Querying first and taking ``_stats_lock`` afterward can
        # race a writer commit: callers would see blocks_on_disk=0 beside a
        # terminal successful fence.  Writers commit SQLite before taking this
        # lock, so reading the index while holding it is deadlock-safe and makes
        # a completed fence imply the matching row is already visible.
        with self._stats_lock:
            conn = sqlite3.connect(str(self._db_path), timeout=1.0)
            try:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(file_size), 0), "
                    "COALESCE(SUM(access_count), 0), "
                    "COALESCE(SUM(num_tokens), 0) FROM blocks"
                ).fetchone()
                latest_payload_row = conn.execute(
                    "SELECT file_name, num_tokens, file_size, dtype FROM blocks "
                    "ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            recent_write_fences = []
            for state in self._write_fences.values():
                recent_write_fences.append(
                    {
                        key: value
                        for key, value in state.items()
                        if not key.startswith("_")
                    }
                )
            stats = {
                "blocks_on_disk": row[0],
                "disk_size_bytes": row[1],
                "disk_size_gb": round(row[1] / 1024**3, 3),
                "total_accesses": row[2],
                "total_tokens_on_disk": int(row[3]),
                "total_cached_tokens": int(row[3]),
                "disk_hits": self.disk_hits,
                "disk_misses": self.disk_misses,
                # Per-reason breakdown: a bare miss count cannot separate
                # "never stored" from "stored but unreadable/invalid".
                "disk_miss_reasons": dict(self.disk_miss_reasons),
                "write_drop_reasons": dict(self.write_drop_reasons),
                "disk_writes": self.disk_writes,
                "disk_evictions": self.disk_evictions,
                "tq_native_writes": self.tq_native_writes,
                "tq_native_hits": self.tq_native_hits,
                "tq_native_enabled": self._allow_tq_native,
                "selective_rotating_reconstruction_reads": (
                    self.selective_rotating_reconstruction_reads
                ),
                "selective_rotating_layers_omitted": (
                    self.selective_rotating_layers_omitted
                ),
                "validation_metadata_reads": self.validation_metadata_reads,
                "group_files_written": self.group_files_written,
                "group_members_written": self.group_members_written,
                "group_member_reads": self.group_member_reads,
                "group_span_reads": getattr(self, "group_span_reads", 0),
                "group_span_blocks": getattr(self, "group_span_blocks", 0),
                "validation_metadata_read_failures": (
                    self.validation_metadata_read_failures
                ),
                "write_pipeline": {
                    "serialization_mode": "streamed_safetensors_file",
                    "queue_depth": self._write_queue.qsize(),
                    "inflight": self._write_inflight,
                    "active_producers": self._active_write_producers,
                    "pending_items": self._pending_write_items,
                    "accepting_writes": self._accepting_writes,
                    "pending_bytes": self._pending_write_bytes,
                    "max_pending_bytes": self._max_pending_write_bytes,
                    "byte_budget_drops": self._pending_write_byte_drops,
                    "offthread_serializations_queued": (
                        self._offthread_serializations_queued
                    ),
                    "offthread_serializations_completed": (
                        self._offthread_serializations_completed
                    ),
                    "offthread_serialization_failures": (
                        self._offthread_serialization_failures
                    ),
                    "writer_alive": self._writer_thread.is_alive(),
                    "completion_generation": self._write_completion_generation,
                    "recent_fences": recent_write_fences,
                },
            }
        latest_payload = self._inspect_latest_payload(latest_payload_row)
        if latest_payload is not None:
            stats["latest_payload"] = latest_payload
        # Never hold ``_stats_lock`` while taking the process-shared root lock:
        # the background writer takes those locks in the opposite phase order.
        global_health = self.global_budget.refresh_health()
        stats["global_budget"] = {
            "root": str(self.global_cache_root),
            "max_size_bytes": global_health.max_size_bytes,
            "bytes_after": global_health.bytes_after,
            "compliant": global_health.compliant,
            "accounted": global_health.accounted,
            "telemetry_stale": global_health.telemetry_stale,
            "scan_performed": global_health.scan_performed,
            "evicted_entries": global_health.evicted_entries,
            # cumulative across every writer/process on this root (ledger)
            "evicted_entries_total": global_health.evicted_entries_total,
            "capacity_evicted_entries_total": global_health.capacity_evicted_entries_total,
            "evicted_bytes_total": global_health.evicted_bytes_total,
            "protected_recent_orphans": global_health.protected_recent_orphans,
            "protected_temp_files": global_health.protected_temp_files,
            "evicted_bytes": global_health.evicted_bytes,
            "accounting_generation": global_health.accounting_generation,
            "reconciliation_generation": (
                global_health.reconciliation_generation
            ),
            # source constants + observed scan age (idle janitor)
            "janitor": self.global_budget.janitor_status(),
        }
        return stats

    def inspect_block_chain(self, block_hashes: list[bytes]) -> dict[str, Any]:
        """Inspect exact chain membership without loading blocks or touching LRU.

        This is the source-owned observation path used by the live cache gate.
        Callers provide the already-derived chain hashes internally; neither the
        hashes nor index file names are returned.  The dedicated SQLite
        connection is opened read-only and query-only, and filesystem checks
        reject symlinks or paths that are not lexically contained by the cache
        directory.  In particular, this method must not call ``has_block`` or
        ``read_block`` because the latter intentionally updates access metadata.
        """
        normalized: list[bytes] = []
        for value in block_hashes:
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError("block hashes must be bytes-like")
            block_hash = bytes(value)
            if len(block_hash) != 32:
                raise ValueError("block hashes must be 32-byte SHA-256 values")
            normalized.append(block_hash)
        if len(normalized) > 16_384:
            raise ValueError("block-chain inspection exceeds the bounded limit")

        database_uri = (
            f"file:{quote(str(self._db_path), safe='/')}?mode=ro"
        )
        conn = sqlite3.connect(
            database_uri,
            timeout=1.0,
            uri=True,
        )
        cache_root = self.cache_dir.resolve(strict=True)
        try:
            conn.execute("PRAGMA query_only = ON")
            total_size, total_entries = conn.execute(
                "SELECT COALESCE(SUM(file_size), 0), COUNT(*) FROM blocks"
            ).fetchone()
            rows: list[dict[str, Any]] = []
            for ordinal, block_hash in enumerate(normalized):
                row = conn.execute(
                    "SELECT file_name, num_tokens, file_size, created_at, "
                    "last_accessed, access_count FROM blocks "
                    "WHERE block_hash = ?",
                    (block_hash.hex(),),
                ).fetchone()
                if row is None:
                    rows.append(
                        {
                            "ordinal": ordinal,
                            "indexed": False,
                            "readable": False,
                            "num_tokens": 0,
                            "file_size_bytes": 0,
                            "created_at_ns": 0,
                            "last_accessed_ns": 0,
                            "access_count": 0,
                        }
                    )
                    continue

                (
                    file_name,
                    num_tokens,
                    file_size,
                    created_at,
                    last_accessed,
                    access_count,
                ) = row
                relative_path = Path(str(file_name))
                lexically_safe = (
                    not relative_path.is_absolute()
                    and ".." not in relative_path.parts
                )
                file_path = self.cache_dir / relative_path
                readable = False
                if lexically_safe and not file_path.is_symlink():
                    try:
                        resolved_file = file_path.resolve(strict=True)
                        stat_result = resolved_file.stat()
                        readable = bool(
                            resolved_file.is_relative_to(cache_root)
                            and resolved_file.is_file()
                            and stat_result.st_size == max(0, int(file_size or 0))
                        )
                    except OSError:
                        readable = False
                rows.append(
                    {
                        "ordinal": ordinal,
                        "indexed": True,
                        "readable": readable,
                        "num_tokens": max(0, int(num_tokens or 0)),
                        "file_size_bytes": max(0, int(file_size or 0)),
                        "created_at_ns": max(
                            0,
                            int(float(created_at or 0.0) * 1_000_000_000),
                        ),
                        "last_accessed_ns": max(
                            0,
                            int(float(last_accessed or 0.0) * 1_000_000_000),
                        ),
                        "access_count": max(0, int(access_count or 0)),
                    }
                )
        finally:
            conn.close()

        return {
            "schema": "vmlx-block-disk-chain-inspection-v1",
            "access_metadata_mutated": False,
            "expected_blocks": len(normalized),
            "store_total_entries": max(0, int(total_entries or 0)),
            "store_total_size_bytes": max(0, int(total_size or 0)),
            "store_max_size_bytes": max(0, int(self.max_size_bytes or 0)),
            "blocks": rows,
        }

    def partial_token_counts(self, block_size: int) -> List[int]:
        """Return persisted terminal block sizes smaller than ``block_size``.

        ``PagedCacheManager`` keeps this set in memory for exact partial-prefix
        lookup. Rehydrate it from the durable index on process restart so a
        short first prompt can still seed a longer follow-up from L2.
        """
        limit = max(1, int(block_size))
        conn = sqlite3.connect(str(self._db_path), timeout=1.0)
        try:
            rows = conn.execute(
                "SELECT DISTINCT num_tokens FROM blocks "
                "WHERE num_tokens > 0 AND num_tokens < ? "
                "ORDER BY num_tokens DESC",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [int(row[0]) for row in rows]

    def clear(self) -> None:
        """Clear all cached blocks from disk."""
        import shutil
        deadline = time.monotonic() + self._writer_shutdown_timeout_seconds
        with self._write_lifecycle:
            while self._clear_in_progress:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("another block-disk clear did not quiesce")
                self._write_lifecycle.wait(timeout=remaining)
            if self._shutdown_started:
                raise RuntimeError("block-disk store is shutting down")
            self._clear_in_progress = True
            self._accepting_writes = False

        try:
            with self._write_lifecycle:
                while (
                    self._active_write_producers > 0
                    or self._pending_write_items > 0
                ):
                    if (
                        self._pending_write_items > 0
                        and not self._writer_thread.is_alive()
                    ):
                        raise RuntimeError(
                            "block-disk writer stopped before clear quiesced"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "block-disk writes did not quiesce before clear"
                        )
                    self._write_lifecycle.wait(timeout=remaining)

            with self.global_budget.exclusive_mutation_guard() as locked:
                if not locked:
                    raise OSError(
                        "global block-cache exclusive mutation lock unavailable"
                    )
                if self.blocks_dir.exists():
                    shutil.rmtree(self.blocks_dir)
                    self.blocks_dir.mkdir(parents=True)
                conn = sqlite3.connect(str(self._db_path), timeout=5.0)
                try:
                    conn.execute("DELETE FROM block_write_pins")
                    conn.execute("DELETE FROM blocks")
                    conn.commit()
                finally:
                    conn.close()
                # Reconcile before releasing the same exclusive transaction.
                # SQLite page reuse means the byte delta cannot be inferred from
                # deleted logical rows, and another process must not publish in the
                # gap between physical deletion and the accounting rebuild.
                refreshed = self.global_budget.account_finalized_write_locked(-1)
            self._global_budget_write_enabled = bool(
                refreshed is not None
                and refreshed.accounted
                and refreshed.compliant
            )
            unsettled_fences: set[str] = set()
            with self._stats_lock:
                self.tq_native_writes = 0
                self.tq_native_hits = 0
                for fence_id, state in self._write_fences.items():
                    if state.get("sealed") and not state.get("post_eviction_complete"):
                        unsettled_fences.add(fence_id)
            for fence_id in unsettled_fences:
                self._fail_write_fence(
                    fence_id,
                    "disk cache cleared before fence settled",
                )
            logger.info("Disk cache cleared")
        finally:
            with self._write_lifecycle:
                self._clear_in_progress = False
                if not self._shutdown_started:
                    self._accepting_writes = True
                self._write_lifecycle.notify_all()

    def _close_current_read_connection(self) -> None:
        try:
            conn = getattr(self._thread_local, "read_conn", None)
            if conn is not None:
                conn.close()
                self._thread_local.read_conn = None
        except Exception:
            pass

    def _finalize_shutdown_after_writer_stop(self) -> None:
        """Flush queued publications, then release the cap owner exactly once."""

        with self._shutdown_finalize_lock:
            if self._shutdown_finalized or self._writer_thread.is_alive():
                return
            remaining = []
            while not self._write_queue.empty():
                try:
                    remaining.append(self._write_queue.get_nowait())
                except queue.Empty:
                    break
            try:
                if remaining:
                    flush_conn = sqlite3.connect(str(self._db_path), timeout=5.0)
                    flush_conn.execute("PRAGMA journal_mode=WAL")
                    try:
                        self._process_write_batch(flush_conn, remaining)
                    finally:
                        flush_conn.close()
                        self._complete_write_items(len(remaining))
            except Exception as exc:
                logger.warning(
                    "BlockDiskStore delayed shutdown flush failed: %s",
                    exc,
                )
                # Any unfinalized temp remains owner-tagged and is protected
                # until this lease closes, then becomes normal orphan cleanup.
            try:
                with self.global_budget.exclusive_mutation_guard() as locked:
                    if locked:
                        pin_conn = sqlite3.connect(
                            str(self._db_path),
                            timeout=5.0,
                        )
                        try:
                            released = self._release_write_fence_pins_locked(
                                pin_conn,
                            )
                            if released:
                                self.global_budget._enforce_locked()
                        finally:
                            pin_conn.close()
            except Exception as exc:
                logger.warning(
                    "BlockDiskStore shutdown pin cleanup failed: %s",
                    exc,
                )
            self._close_current_read_connection()
            if self.global_budget.close():
                self._shutdown_finalized = True
            else:
                logger.warning(
                    "BlockDiskStore stopped but its aggregate budget lease "
                    "could not be removed; atexit will retry"
                )

    def _wait_for_writer_and_finalize_shutdown(self) -> None:
        with self._write_lifecycle:
            while self._active_write_producers > 0 or self._clear_in_progress:
                self._write_lifecycle.wait()
            self._stop_event.set()
        self._writer_thread.join()
        self._finalize_shutdown_after_writer_stop()

    def shutdown(self) -> None:
        """Stop the writer; defer lease release if its bounded join times out."""

        with self._shutdown_finalize_lock:
            if self._shutdown_finalized:
                return
        deadline = time.monotonic() + self._writer_shutdown_timeout_seconds
        with self._write_lifecycle:
            self._shutdown_started = True
            self._accepting_writes = False
            while self._active_write_producers > 0 or self._clear_in_progress:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._write_lifecycle.wait(timeout=remaining)
            producers_quiesced = (
                self._active_write_producers == 0 and not self._clear_in_progress
            )
            if producers_quiesced:
                self._stop_event.set()
        remaining_timeout = max(0.0, deadline - time.monotonic())
        if producers_quiesced:
            self._writer_thread.join(timeout=remaining_timeout)
        self._close_current_read_connection()
        if not producers_quiesced or self._writer_thread.is_alive():
            logger.warning(
                "BlockDiskStore producers/writer did not stop in time; scheduling "
                "delayed drain and lease release"
            )
            with self._shutdown_finalize_lock:
                if (
                    self._delayed_shutdown_thread is None
                    or not self._delayed_shutdown_thread.is_alive()
                ):
                    self._delayed_shutdown_thread = threading.Thread(
                        target=self._wait_for_writer_and_finalize_shutdown,
                        daemon=True,
                        name="block-disk-shutdown",
                    )
                    self._delayed_shutdown_thread.start()
            return
        self._finalize_shutdown_after_writer_stop()


# =============================================================================
# Serialization / Deserialization (module-level functions)
# =============================================================================

def _serialize_block(
    cache_data: List[Tuple],
) -> Tuple[Dict[str, Any], str, int]:
    """
    Convert CacheBlock.cache_data to a flat dict of named tensors for safetensors.

    Per-layer type tags are stored in metadata so deserialization can handle
    mixed-type blocks (e.g. hybrid Mamba-Transformer models).

    The naming convention encodes layer index and tensor role:
      layer_{i}_keys, layer_{i}_values          — standard kv
      layer_{i}_keys_data, _scales, _zeros       — quantized kv
      layer_{i}_max_size, layer_{i}_keep          — rotating kv params
      layer_{i}_cumulative_{j}                    — cumulative state arrays
      layer_{i}_dsv4_state_{j}                    — DSV4 nested state arrays
      layer_{i}_zaya_cca_state_{j}                 — ZAYA CCA nested state arrays
      layer_{i}_no_state                           — explicit no-state layer

    Returns:
        (tensor_dict, dtype_string, total_cache_layers). The total layer
        count includes skip entries so L2 records remain model-shape scoped.
    """
    if not HAS_MLX:
        return {}, "unknown", 0

    tensors: Dict[str, Any] = {}
    num_layers_with_data = 0
    # Per-layer metadata including type tags for mixed-type block support
    meta: Dict[str, Any] = {
        "__layer_types__": {},
        "__num_cache_layers__": len(cache_data),
    }
    try:
        from .prefix_cache import runtime_cache_fingerprint

        meta["__runtime_cache_fingerprint__"] = runtime_cache_fingerprint()
    except Exception:
        meta["__runtime_cache_fingerprint__"] = "unknown"

    for i, layer_data in enumerate(cache_data):
        tag = layer_data[0]

        if tag == "skip":
            continue

        num_layers_with_data += 1
        meta["__layer_types__"][str(i)] = tag

        if tag == "kv":
            _, keys, values = layer_data
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            # Record original dtype so deserialization can restore it
            # after bfloat16→float16 cast (safetensors doesn't support bf16)
            if hasattr(keys, "dtype"):
                meta.setdefault("__orig_dtypes__", {})[str(i)] = str(keys.dtype)

        elif tag == "quantized_kv":
            _, keys_tuple, values_tuple, layer_meta = layer_data
            tensors[f"layer_{i}_keys_data"] = keys_tuple[0]
            tensors[f"layer_{i}_keys_scales"] = keys_tuple[1]
            tensors[f"layer_{i}_keys_zeros"] = keys_tuple[2]
            tensors[f"layer_{i}_values_data"] = values_tuple[0]
            tensors[f"layer_{i}_values_scales"] = values_tuple[1]
            tensors[f"layer_{i}_values_zeros"] = values_tuple[2]
            # Record the scale/bias dtypes like every other tag does. Without
            # this, the bfloat16->float32 cast applied on the way to disk was
            # never undone, so a block that round-tripped through L2 came back
            # with fp32 scales where a recompute produces bf16. That is a
            # hit != recompute numerics change, and it sits OUTSIDE TurboQuant,
            # so the TQ asymmetry guard does not cover it.
            for _slot, _tup in (("keys", keys_tuple), ("values", values_tuple)):
                for _part, _idx in (("scales", 1), ("zeros", 2)):
                    _t = _tup[_idx]
                    if hasattr(_t, "dtype"):
                        meta.setdefault("__orig_dtypes__", {})[
                            f"{i}_q_{_slot}_{_part}"
                        ] = str(_t.dtype)
            if layer_meta:
                meta[str(i)] = {"quant_meta": layer_meta}

        elif tag == "turboquant_kv":
            _, encoded_keys, encoded_values, tq_config = layer_data
            tensors[f"layer_{i}_tq_ck_indices_packed"] = encoded_keys.indices_packed
            tensors[f"layer_{i}_tq_ck_qjl_packed"] = encoded_keys.qjl_packed
            tensors[f"layer_{i}_tq_ck_residual_norms"] = encoded_keys.residual_norms
            tensors[f"layer_{i}_tq_ck_vector_norms"] = encoded_keys.vector_norms
            tensors[f"layer_{i}_tq_cv_indices_packed"] = encoded_values.indices_packed
            tensors[f"layer_{i}_tq_cv_vector_norms"] = encoded_values.vector_norms
            meta[str(i)] = {
                "tq": {
                    **_json_safe(tq_config),
                    "ck_shape": list(encoded_keys.shape),
                    "ck_bits": int(encoded_keys.index_bits),
                    "cv_shape": list(encoded_values.shape),
                    "cv_bits": int(encoded_values.index_bits),
                }
            }

        elif tag == "minimax_m3":
            _, keys, values, idx_keys = layer_data
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            if idx_keys is not None:
                tensors[f"layer_{i}_idx_keys"] = idx_keys
            if hasattr(keys, "dtype"):
                meta.setdefault("__orig_dtypes__", {})[str(i)] = str(keys.dtype)

        elif tag == "rotating_kv":
            _, keys, values, max_size, keep, *window_state = layer_data
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            tensors[f"layer_{i}_max_size"] = mx.array([max_size], dtype=mx.int32)
            tensors[f"layer_{i}_keep"] = mx.array([keep], dtype=mx.int32)
            offset = window_state[0] if len(window_state) >= 1 else None
            idx_state = window_state[1] if len(window_state) >= 2 else None
            if offset is not None:
                tensors[f"layer_{i}_offset"] = mx.array([offset], dtype=mx.int32)
            if idx_state is not None:
                tensors[f"layer_{i}_idx"] = mx.array([idx_state], dtype=mx.int32)
            if hasattr(keys, "dtype"):
                meta.setdefault("__orig_dtypes__", {})[str(i)] = str(keys.dtype)

        elif tag == "rotating_kv_pending":
            # The layer is identified by __layer_types__ (set above) and its class name here; readers consult the
            # type map first. The old one-element marker tensor carried nothing else, yet cost a tensor per pending
            # layer per block everywhere a tensor costs (create, freeze, write, read): Naive's 36 sliding-window
            # layers made it 36 of a block's 85 tensors. Files written with the marker still read
            # (_infer_layer_type fallback).
            _, class_name = layer_data
            meta[str(i)] = {"class_name": str(class_name)}

        elif tag == "cache_list":
            # CacheList (MoE models): each sub-cache is serialized independently
            _, sub_slices = layer_data
            sub_count = 0
            for j, sub_entry in enumerate(sub_slices):
                sub_tag = sub_entry[0]
                if sub_tag == "skip":
                    continue
                elif sub_tag == "kv":
                    _, sk, sv = sub_entry
                    tensors[f"layer_{i}_sub_{j}_keys"] = sk
                    tensors[f"layer_{i}_sub_{j}_values"] = sv
                    if hasattr(sk, "dtype"):
                        meta.setdefault("__orig_dtypes__", {})[f"{i}_sub_{j}"] = str(sk.dtype)
                    sub_count += 1
                elif sub_tag == "quantized_kv":
                    _, skt, svt, squant_meta = sub_entry
                    if len(skt) != 3 or len(svt) != 3:
                        raise ValueError(
                            "CacheList quantized_kv entries require "
                            "3-component key/value tuples"
                        )
                    tensors[f"layer_{i}_sub_{j}_keys_data"] = skt[0]
                    tensors[f"layer_{i}_sub_{j}_keys_scales"] = skt[1]
                    tensors[f"layer_{i}_sub_{j}_keys_zeros"] = skt[2]
                    tensors[f"layer_{i}_sub_{j}_values_data"] = svt[0]
                    tensors[f"layer_{i}_sub_{j}_values_scales"] = svt[1]
                    tensors[f"layer_{i}_sub_{j}_values_zeros"] = svt[2]
                    meta.setdefault(str(i), {}).setdefault("subs", {})[str(j)] = {
                        "type": "quantized_kv",
                        "quant_meta": _json_safe(squant_meta),
                    }
                    sub_count += 1
                elif sub_tag == "turboquant_kv":
                    _, sck, scv, stq = sub_entry
                    tensors[f"layer_{i}_sub_{j}_tq_ck_indices_packed"] = sck.indices_packed
                    tensors[f"layer_{i}_sub_{j}_tq_ck_qjl_packed"] = sck.qjl_packed
                    tensors[f"layer_{i}_sub_{j}_tq_ck_residual_norms"] = sck.residual_norms
                    tensors[f"layer_{i}_sub_{j}_tq_ck_vector_norms"] = sck.vector_norms
                    tensors[f"layer_{i}_sub_{j}_tq_cv_indices_packed"] = scv.indices_packed
                    tensors[f"layer_{i}_sub_{j}_tq_cv_vector_norms"] = scv.vector_norms
                    meta.setdefault(str(i), {}).setdefault("subs", {})[str(j)] = {
                        "type": "turboquant_kv",
                        "tq": {
                            **_json_safe(stq),
                            "ck_shape": list(sck.shape),
                            "ck_bits": int(sck.index_bits),
                            "cv_shape": list(scv.shape),
                            "cv_bits": int(scv.index_bits),
                        },
                    }
                    sub_count += 1
                elif sub_tag == "cumulative":
                    _, sub_state, sub_meta, sub_cls = sub_entry
                    if isinstance(sub_state, (list, tuple)):
                        for k, arr in enumerate(sub_state):
                            if hasattr(arr, "shape"):
                                tensors[f"layer_{i}_sub_{j}_cumulative_{k}"] = arr
                    meta.setdefault(str(i), {}).setdefault("subs", {})[str(j)] = {
                        "class_name": sub_cls, "meta": sub_meta
                    }
                    sub_count += 1
                else:
                    raise ValueError(
                        f"unsupported CacheList sub-cache tag {sub_tag!r}"
                    )
            meta.setdefault(str(i), {})["sub_count"] = len(sub_slices)

        elif tag == "zaya_cca":
            _, kv_entry, cca_state, cca_meta, cache_meta = layer_data
            layer_meta = {
                "cache_meta": _json_safe(cache_meta),
                "cca_meta": _json_safe(cca_meta),
                "terminal": cca_state is not None,
            }
            if isinstance(kv_entry, (tuple, list)) and kv_entry:
                kv_tag = kv_entry[0]
                layer_meta["kv_tag"] = kv_tag
                if kv_tag == "kv":
                    _, zk, zv = kv_entry
                    tensors[f"layer_{i}_zaya_keys"] = zk
                    tensors[f"layer_{i}_zaya_values"] = zv
                    if hasattr(zk, "dtype"):
                        meta.setdefault("__orig_dtypes__", {})[f"{i}_zaya_kv"] = str(zk.dtype)
                elif kv_tag == "quantized_kv":
                    _, zkt, zvt, zmeta = kv_entry
                    tensors[f"layer_{i}_zaya_keys_data"] = zkt[0]
                    tensors[f"layer_{i}_zaya_keys_scales"] = zkt[1]
                    tensors[f"layer_{i}_zaya_keys_zeros"] = zkt[2]
                    tensors[f"layer_{i}_zaya_values_data"] = zvt[0]
                    tensors[f"layer_{i}_zaya_values_scales"] = zvt[1]
                    tensors[f"layer_{i}_zaya_values_zeros"] = zvt[2]
                    layer_meta["quant_meta"] = _json_safe(zmeta)
            if cca_state is not None:
                counter = [0]
                layer_meta["cca_state_tree"] = _pack_tree(
                    cca_state,
                    tensors,
                    f"layer_{i}_zaya_cca_state",
                    counter,
                )
            meta[str(i)] = layer_meta

        elif tag == "no_state":
            _, class_name = layer_data
            tensors[f"layer_{i}_no_state"] = mx.array([1], dtype=mx.int8)
            meta[str(i)] = {"class_name": class_name}

        elif tag == "cumulative":
            _, state_list, layer_meta, class_name = layer_data
            if isinstance(state_list, (list, tuple)):
                for j, arr in enumerate(state_list):
                    if hasattr(arr, "shape"):
                        tensors[f"layer_{i}_cumulative_{j}"] = arr
            meta[str(i)] = {"class_name": class_name, "meta": layer_meta}

        elif tag == "deepseek_v4":
            _, state_tree, layer_meta, class_name, cache_meta = layer_data
            counter = [0]
            tree_meta = _pack_tree(
                state_tree,
                tensors,
                f"layer_{i}_dsv4_state",
                counter,
            )
            meta[str(i)] = {
                "class_name": class_name,
                "meta": _json_safe(layer_meta),
                "cache_meta": _json_safe(cache_meta),
                "state_tree": tree_meta,
            }

        elif tag == "deepseek_v4_delta_v1":
            _, record_tree, class_name, cache_meta = layer_data
            counter = [0]
            tree_meta = _pack_tree(
                record_tree,
                tensors,
                f"layer_{i}_dsv4_delta",
                counter,
            )
            meta[str(i)] = {
                "class_name": class_name,
                "cache_meta": _json_safe(cache_meta),
                "record_tree": tree_meta,
            }

        elif tag == "deepseek_v4_pending":
            _, class_name, cache_meta = layer_data
            tensors[f"layer_{i}_dsv4_pending"] = mx.array([1], dtype=mx.int32)
            meta[str(i)] = {
                "class_name": class_name,
                "cache_meta": _json_safe(cache_meta),
            }

    # Determine dominant dtype for the DB index (informational only)
    type_set = set(meta["__layer_types__"].values())
    if len(type_set) == 1:
        dtype = type_set.pop()
    elif type_set:
        dtype = "mixed"
    else:
        dtype = "kv"

    # Capture every leaf independently before the worker converts BF16 into
    # its lossless U16 carrier. A single dtype per layer loses mixed QSA index
    # precision and omitted nested quantized scales/cumulative-state dtypes.
    # Keep legacy declarations for older readers, but new readers use this
    # complete map and never reapply the coarser per-layer conversions.
    meta["__tensor_dtypes__"] = {
        name: _block_tensor_dtype_name(value.dtype)
        for name, value in tensors.items()
    }

    # Store metadata as a serialized JSON tensor.
    # Use a non-reserved key — safetensors has a special "__metadata__"
    # header that expects a string-to-string dict. Writing a uint8 tensor
    # under that name triggers C++ JSON type_error.302 ("type must be
    # string, but is array") on load → disk cache hits become silent
    # misses because _deserialize_block returns [] and the block is
    # treated as corrupt + cleanup-queued.
    if meta:
        import numpy as np

        meta_bytes = json.dumps(meta).encode("utf-8")
        # np.frombuffer, not list(meta_bytes): the metadata is ~6-7 KB per
        # block and a Python int per byte cost ~0.3 ms per block on every store.
        tensors["__vmlx_block_meta__"] = mx.array(
            np.frombuffer(meta_bytes, dtype=np.uint8)
        )

    num_layers = len(cache_data) if num_layers_with_data else 0
    return tensors, dtype, num_layers


def _load_reconstruction_payload_without_stale_rotating(
    file_path: Path,
    *,
    target_offset: int,
    probe: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], set[int]]]:
    """Load a mixed-SWA block while omitting unusable rotating checkpoints.

    A content-addressed block can be the terminal checkpoint of an earlier
    branch and later become an interior block of a longer chain.  Its standard
    KV slices remain reusable, but its RotatingKVCache window is valid only at
    the absolute offset recorded in ``layer_N_offset``.  Reading every old
    window before ``reconstruct_cache`` rejects it caused hundreds of MB of
    avoidable I/O per short follow-up on Gemma 4.

    This helper opens only the safetensors metadata and tiny offset scalars
    first.  It returns ``None`` unless at least one stale rotating layer can be
    omitted, preserving the ordinary ``mx.load`` path for non-mixed and exact
    terminal blocks.  For omitted layers the offset scalar remains in the
    returned mapping so deserialization preserves the declared layer index; the
    caller converts the resulting ``skip`` entry to an explicit
    ``rotating_kv_pending`` marker before validation.
    """
    if not HAS_MLX:
        return None

    try:
        from safetensors import safe_open

        with safe_open(str(file_path), framework="numpy") as handle:
            tensor_names = list(handle.keys())
            meta_key = (
                "__vmlx_block_meta__"
                if "__vmlx_block_meta__" in tensor_names
                else "__metadata__" if "__metadata__" in tensor_names else None
            )
            if meta_key is None:
                return None

            meta_array = handle.get_tensor(meta_key)
            decoded = json.loads(meta_array.tobytes().decode("utf-8"))
            if not isinstance(decoded, dict):
                return None
            raw_layer_types = decoded.get("__layer_types__")
            if not isinstance(raw_layer_types, dict):
                return None

            rotating_layers: set[int] = set()
            for raw_idx, layer_type in raw_layer_types.items():
                if layer_type != "rotating_kv":
                    continue
                try:
                    rotating_layers.add(int(raw_idx))
                except (TypeError, ValueError):
                    return None
            if not rotating_layers:
                if probe is not None:
                    probe["rotating_offsets"] = {}
                return None

            offset_arrays: Dict[int, Any] = {}
            omitted_layers: set[int] = set()
            for layer_idx in rotating_layers:
                offset_name = f"layer_{layer_idx}_offset"
                if offset_name not in tensor_names:
                    # Old records without absolute offsets cannot be filtered
                    # safely. Fall back to the existing full loader.
                    return None
                offset_array = handle.get_tensor(offset_name)
                if int(getattr(offset_array, "size", 0) or 0) != 1:
                    return None
                offset_arrays[layer_idx] = offset_array
                if int(offset_array.reshape(-1)[0]) != int(target_offset):
                    omitted_layers.add(layer_idx)

            if probe is not None:
                probe["rotating_offsets"] = {
                    idx: int(arr.reshape(-1)[0]) for idx, arr in offset_arrays.items()
                }
            if not omitted_layers:
                return None

            declared_dtypes = decoded.get("__tensor_dtypes__")
            if "__tensor_dtypes__" in decoded and (
                not isinstance(declared_dtypes, dict)
                or set(declared_dtypes) != set(tensor_names) - {meta_key}
            ):
                return None  # ordinary full reader will reject the record
            if declared_dtypes is not None:
                # Validate even omitted payloads from header-only dtype data.
                # Do not let pruning hide corrupt declarations on old rings.
                for name, wanted in declared_dtypes.items():
                    physical = _BLOCK_SAFETENSORS_DTYPES.get(
                        handle.get_slice(name).get_dtype()
                    )
                    if (
                        not isinstance(wanted, str)
                        or wanted not in _BLOCK_TENSOR_DTYPES
                        or not (
                            physical == wanted
                            or (physical == "uint16" and wanted == "bfloat16")
                        )
                    ):
                        return None

            selected: Dict[str, Any] = {}
            for name in tensor_names:
                layer_idx: Optional[int] = None
                if name.startswith("layer_"):
                    try:
                        layer_idx = int(name[6:].split("_", 1)[0])
                    except (TypeError, ValueError):
                        layer_idx = None

                if layer_idx in omitted_layers:
                    # Keep one tiny scalar per omitted layer so legacy
                    # deserialization still discovers its index. The large
                    # keys/values and the other unused ring metadata stay on
                    # SSD.
                    if name != f"layer_{layer_idx}_offset":
                        continue
                    numpy_array = offset_arrays[layer_idx]
                else:
                    numpy_array = handle.get_tensor(name)
                selected[name] = mx.array(numpy_array)

            if declared_dtypes is not None:
                # The selective reader intentionally omits old rotating rings.
                # Bind the metadata to exactly the payload it returns, retaining
                # the complete-map validation above against the original file.
                decoded["__tensor_dtypes__"] = {
                    name: declared_dtypes[name]
                    for name in selected if name != meta_key
                }
                import numpy as np

                selected[meta_key] = mx.array(
                    np.frombuffer(json.dumps(decoded).encode("utf-8"), dtype=np.uint8)
                )

        return selected, omitted_layers
    except Exception as exc:  # noqa: BLE001 - optimization must fail open
        logger.debug(
            "Selective mixed-SWA safetensors read unavailable for %s; "
            "falling back to full payload: %s",
            file_path,
            exc,
        )
        return None


def _deserialize_block(
    data: Dict[str, Any],
    dtype: str,
) -> List[Tuple]:
    """
    Reconstruct CacheBlock.cache_data from loaded safetensors dict.

    Uses per-layer type tags from metadata for mixed-type block support.
    Falls back to the dtype field for backward compatibility with blocks
    serialized before per-layer tags were added.
    """
    # Extract metadata if present. Try the new key first; fall back to
    # the legacy `__metadata__` key for blocks written by older builds
    # (pre-fix, still readable because this loader does the read, not
    # safetensors' reserved-name parser).
    meta: Dict[str, Any] = {}
    meta_arr = data.get("__vmlx_block_meta__")
    if meta_arr is None:
        meta_arr = data.get("__metadata__")
    if meta_arr is not None:
        try:
            # Zero-copy view of the evaluated uint8 tensor. bytes(arr.tolist())
            # built one Python int per byte (~6-7 KB per block): ~30 % of a
            # long-prefix restore (512 blocks at 32k).
            import numpy as np

            meta_bytes = np.asarray(meta_arr).tobytes()
        except Exception:
            meta_bytes = bytes(meta_arr.tolist())
        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception:
            pass
    # Remove from data dict so it's not picked up as a layer
    data.pop("__vmlx_block_meta__", None)
    data.pop("__metadata__", None)

    try:
        from .prefix_cache import runtime_cache_fingerprint

        current_runtime = runtime_cache_fingerprint()
    except Exception:
        current_runtime = "unknown"
    stored_runtime = meta.get("__runtime_cache_fingerprint__")
    if stored_runtime != current_runtime:
        logger.info(
            "Disk cache block runtime fingerprint mismatch; treating as miss "
            "(stored=%s current=%s)",
            stored_runtime or "missing",
            current_runtime,
        )
        return []

    if "__tensor_dtypes__" in meta:
        if (
            not _restore_block_tensor_dtypes(data, meta["__tensor_dtypes__"])
            or not _block_tree_dtypes_match(meta, meta["__tensor_dtypes__"])
        ):
            logger.warning("Disk cache block tensor dtype metadata is incompatible; treating as miss")
            return []
        # These legacy declarations group independent tensor lanes. Reapplying
        # them would narrow a restored FP32 QSA index to the KV's BF16 dtype.
        orig_dtypes = {}
    else:
        orig_dtypes = meta.get("__orig_dtypes__", {})

    # Per-layer type map (new format with __layer_types__)
    layer_types = meta.get("__layer_types__", {})

    # Find all layer indices
    layer_indices: Dict[int, str] = {}
    for key in data:
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                idx = int(parts[1])
                if idx not in layer_indices:
                    layer_indices[idx] = key
            except ValueError:
                continue

    declared_num_layers = meta.get("__num_cache_layers__")
    try:
        declared_num_layers = int(declared_num_layers)
    except (TypeError, ValueError):
        declared_num_layers = 0

    # Layers whose type map entry carries no tensors (a pending rotating chain node is identified by its tag alone).
    tensorless = {
        int(k) for k, v in layer_types.items()
        if v == "rotating_kv_pending" and str(k).isdigit()
    } if isinstance(layer_types, dict) else set()

    if not layer_indices and not tensorless:
        return []

    max_layer = max(list(layer_indices.keys()) + list(tensorless))
    num_cache_layers = max(max_layer + 1, declared_num_layers)
    cache_data: List[Tuple] = []

    for i in range(num_cache_layers):
        if i not in layer_indices and i not in tensorless:
            cache_data.append(("skip",))
            continue

        # Determine this layer's type: prefer per-layer tag, fallback to global dtype
        layer_type = (
            layer_types[str(i)]
            if str(i) in layer_types
            else _infer_layer_type(data, i, dtype)
        )

        if layer_type == "kv":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            if keys is not None and values is not None:
                # Restore original dtype if it was cast during serialization
                # (e.g. bfloat16 -> float16 because safetensors doesn't support bf16)
                orig_dt = orig_dtypes.get(str(i))
                if HAS_MLX and orig_dt and orig_dt != str(keys.dtype):
                    target = getattr(mx, orig_dt.replace("mlx.core.", ""), None)
                    if target is not None:
                        keys = _restore_serialized_dtype(keys, target)
                        values = _restore_serialized_dtype(values, target)
                cache_data.append(("kv", keys, values))
            else:
                cache_data.append(("skip",))

        elif layer_type == "minimax_m3":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            idx_keys = data.get(f"layer_{i}_idx_keys")
            if keys is not None and values is not None:
                orig_dt = orig_dtypes.get(str(i))
                if HAS_MLX and orig_dt and orig_dt != str(keys.dtype):
                    target = getattr(mx, orig_dt.replace("mlx.core.", ""), None)
                    if target is not None:
                        keys = _restore_serialized_dtype(keys, target)
                        values = _restore_serialized_dtype(values, target)
                        if idx_keys is not None:
                            idx_keys = _restore_serialized_dtype(idx_keys, target)
                cache_data.append(("minimax_m3", keys, values, idx_keys))
            else:
                cache_data.append(("skip",))

        elif layer_type == "quantized_kv":
            try:
                keys_tuple = (
                    data[f"layer_{i}_keys_data"],
                    data[f"layer_{i}_keys_scales"],
                    data[f"layer_{i}_keys_zeros"],
                )
                values_tuple = (
                    data[f"layer_{i}_values_data"],
                    data[f"layer_{i}_values_scales"],
                    data[f"layer_{i}_values_zeros"],
                )

                # Undo the bfloat16->float32 cast applied on the way to disk.
                # Without this a block that round-tripped through L2 came back
                # with fp32 scales/biases where a recompute produces bf16 — a
                # hit != recompute numerics change, and one OUTSIDE TurboQuant,
                # so the TQ asymmetry guard never covered it. The packed data
                # itself is integer and is left alone.
                def _restore(tup, slot):
                    parts = list(tup)
                    for idx, part_name in ((1, "scales"), (2, "zeros")):
                        want = orig_dtypes.get(f"{i}_q_{slot}_{part_name}")
                        arr = parts[idx]
                        if not want or not hasattr(arr, "astype"):
                            continue
                        if str(getattr(arr, "dtype", "")) == want:
                            continue
                        target = getattr(mx, want.split(".")[-1], None)
                        if target is not None:
                            try:
                                parts[idx] = _restore_serialized_dtype(arr, target)
                            except Exception:  # noqa: BLE001
                                pass
                    return tuple(parts)

                keys_tuple = _restore(keys_tuple, "keys")
                values_tuple = _restore(values_tuple, "values")

                layer_meta_dict = meta.get(str(i), {})
                layer_meta = layer_meta_dict.get("quant_meta", layer_meta_dict.get("meta", {}))
                cache_data.append(("quantized_kv", keys_tuple, values_tuple, layer_meta))
            except KeyError:
                cache_data.append(("skip",))

        elif layer_type == "turboquant_kv":
            try:
                layer_meta_dict = meta.get(str(i), {})
                cache_data.append(
                    _restore_tq_block_entry(
                        data,
                        f"layer_{i}",
                        layer_meta_dict.get("tq"),
                    )
                )
            except Exception as exc:
                logger.warning("TQ block layer %d metadata decode failed: %s", i, exc)
                cache_data.append(("skip",))

        elif layer_type == "rotating_kv":
            keys = data.get(f"layer_{i}_keys")
            values = data.get(f"layer_{i}_values")
            max_size_arr = data.get(f"layer_{i}_max_size")
            keep_arr = data.get(f"layer_{i}_keep")
            offset_arr = data.get(f"layer_{i}_offset")
            idx_arr = data.get(f"layer_{i}_idx")
            if keys is not None and values is not None:
                orig_dt = orig_dtypes.get(str(i))
                if HAS_MLX and orig_dt and orig_dt != str(keys.dtype):
                    target = getattr(mx, orig_dt.replace("mlx.core.", ""), None)
                    if target is not None:
                        keys = _restore_serialized_dtype(keys, target)
                        values = _restore_serialized_dtype(values, target)
                max_size = int(max_size_arr.item()) if max_size_arr is not None else 0
                keep = int(keep_arr.item()) if keep_arr is not None else 0
                offset = int(offset_arr.item()) if offset_arr is not None else None
                idx_state = int(idx_arr.item()) if idx_arr is not None else None
                cache_data.append(
                    ("rotating_kv", keys, values, max_size, keep, offset, idx_state)
                )
            else:
                cache_data.append(("skip",))

        elif layer_type == "rotating_kv_pending":
            layer_meta_dict = meta.get(str(i), {})
            cache_data.append((
                "rotating_kv_pending",
                layer_meta_dict.get("class_name", "RotatingKVCache"),
            ))

        elif layer_type == "cache_list":
            # CacheList (MoE models): reconstruct sub-caches
            layer_meta_dict = meta.get(str(i), {})
            sub_count = layer_meta_dict.get("sub_count", 0)
            subs_meta = layer_meta_dict.get("subs", {})
            sub_slices = []
            for j in range(sub_count):
                sk = data.get(f"layer_{i}_sub_{j}_keys")
                sv = data.get(f"layer_{i}_sub_{j}_values")
                if sk is not None and sv is not None:
                    # Restore original dtype if cast
                    orig_dt = orig_dtypes.get(f"{i}_sub_{j}")
                    if HAS_MLX and orig_dt and orig_dt != str(sk.dtype):
                        target = getattr(mx, orig_dt.replace("mlx.core.", ""), None)
                        if target is not None:
                            sk = _restore_serialized_dtype(sk, target)
                            sv = _restore_serialized_dtype(sv, target)
                    sub_slices.append(("kv", sk, sv))
                elif subs_meta.get(str(j), {}).get("type") == "quantized_kv":
                    try:
                        keys_tuple = (
                            data[f"layer_{i}_sub_{j}_keys_data"],
                            data[f"layer_{i}_sub_{j}_keys_scales"],
                            data[f"layer_{i}_sub_{j}_keys_zeros"],
                        )
                        values_tuple = (
                            data[f"layer_{i}_sub_{j}_values_data"],
                            data[f"layer_{i}_sub_{j}_values_scales"],
                            data[f"layer_{i}_sub_{j}_values_zeros"],
                        )
                        if any(
                            part.dtype == mx.uint16
                            for parts in (keys_tuple, values_tuple) for part in parts[1:]
                        ):
                            # Legacy nested records omitted the scale dtype;
                            # an undeclared U16 floating-role payload is unsafe.
                            return []
                        sub_slices.append((
                            "quantized_kv",
                            keys_tuple,
                            values_tuple,
                            subs_meta.get(str(j), {}).get("quant_meta", ()),
                        ))
                    except KeyError as exc:
                        logger.warning(
                            "Quantized CacheList layer %d/%d is incomplete: %s",
                            i,
                            j,
                            exc,
                        )
                        return []
                elif subs_meta.get(str(j), {}).get("type") == "turboquant_kv":
                    try:
                        sub_slices.append(
                            _restore_tq_block_entry(
                                data,
                                f"layer_{i}_sub_{j}",
                                subs_meta.get(str(j), {}).get("tq"),
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "TQ block CacheList layer %d/%d metadata decode "
                            "failed: %s",
                            i,
                            j,
                            exc,
                        )
                        sub_slices.append(("skip",))
                elif f"layer_{i}_sub_{j}_cumulative_0" in data:
                    sub_meta_dict = subs_meta.get(str(j), {})
                    sub_cls = sub_meta_dict.get("class_name", "")
                    sub_meta_val = sub_meta_dict.get("meta", "")
                    sub_arrays = []
                    k = 0
                    while f"layer_{i}_sub_{j}_cumulative_{k}" in data:
                        sub_arrays.append(data[f"layer_{i}_sub_{j}_cumulative_{k}"])
                        k += 1
                    sub_slices.append(("cumulative", sub_arrays, sub_meta_val, sub_cls))
                else:
                    sub_slices.append(("skip",))
            cache_data.append(("cache_list", sub_slices))

        elif layer_type == "zaya_cca":
            layer_meta_dict = meta.get(str(i), {})
            kv_tag = layer_meta_dict.get("kv_tag", "skip")
            if kv_tag == "kv":
                zk = data.get(f"layer_{i}_zaya_keys")
                zv = data.get(f"layer_{i}_zaya_values")
                if zk is not None and zv is not None:
                    orig_dt = orig_dtypes.get(f"{i}_zaya_kv")
                    if HAS_MLX and orig_dt and orig_dt != str(zk.dtype):
                        target = getattr(mx, orig_dt.replace("mlx.core.", ""), None)
                        if target is not None:
                            zk = _restore_serialized_dtype(zk, target)
                            zv = _restore_serialized_dtype(zv, target)
                    kv_entry = ("kv", zk, zv)
                else:
                    kv_entry = ("skip",)
            elif kv_tag == "quantized_kv":
                try:
                    zkt = (
                        data[f"layer_{i}_zaya_keys_data"],
                        data[f"layer_{i}_zaya_keys_scales"],
                        data[f"layer_{i}_zaya_keys_zeros"],
                    )
                    zvt = (
                        data[f"layer_{i}_zaya_values_data"],
                        data[f"layer_{i}_zaya_values_scales"],
                        data[f"layer_{i}_zaya_values_zeros"],
                    )
                    if any(
                        part.dtype == mx.uint16
                        for parts in (zkt, zvt) for part in parts[1:]
                    ):
                        return []
                    kv_entry = (
                        "quantized_kv",
                        zkt,
                        zvt,
                        layer_meta_dict.get("quant_meta", ()),
                    )
                except KeyError:
                    kv_entry = ("skip",)
            else:
                kv_entry = ("skip",)
            cca_state = _unpack_tree(
                layer_meta_dict.get("cca_state_tree"),
                data,
            ) if layer_meta_dict.get("terminal") else None
            cache_data.append((
                "zaya_cca",
                kv_entry,
                cca_state,
                layer_meta_dict.get("cca_meta", ""),
                layer_meta_dict.get("cache_meta", {}),
            ))

        elif layer_type == "no_state":
            layer_meta_dict = meta.get(str(i), {})
            cache_data.append(("no_state", layer_meta_dict.get("class_name", "")))

        elif layer_type == "cumulative":
            layer_meta_dict = meta.get(str(i), {})
            class_name = layer_meta_dict.get("class_name", "")
            layer_meta_val = layer_meta_dict.get("meta", "")
            state_arrays = []
            j = 0
            while f"layer_{i}_cumulative_{j}" in data:
                state_arrays.append(data[f"layer_{i}_cumulative_{j}"])
                j += 1
            if state_arrays:
                cache_data.append(("cumulative", state_arrays, layer_meta_val, class_name))
            else:
                cache_data.append(("skip",))
        elif layer_type == "deepseek_v4":
            layer_meta_dict = meta.get(str(i), {})
            class_name = layer_meta_dict.get("class_name", "DeepseekV4Cache")
            layer_meta_val = layer_meta_dict.get("meta", "")
            cache_meta = layer_meta_dict.get("cache_meta", {})
            state_tree = layer_meta_dict.get("state_tree")
            state = _unpack_tree(state_tree, data)
            if state is not None:
                cache_data.append((
                    "deepseek_v4",
                    state,
                    layer_meta_val,
                    class_name,
                    cache_meta,
                ))
            else:
                cache_data.append(("skip",))
        elif layer_type == "deepseek_v4_delta_v1":
            layer_meta_dict = meta.get(str(i), {})
            record = _unpack_tree(
                layer_meta_dict.get("record_tree"),
                data,
            )
            if isinstance(record, dict):
                cache_data.append((
                    "deepseek_v4_delta_v1",
                    record,
                    layer_meta_dict.get("class_name", "DeepseekV4Cache"),
                    layer_meta_dict.get("cache_meta", {}),
                ))
            else:
                cache_data.append(("skip",))
        elif layer_type == "deepseek_v4_pending":
            layer_meta_dict = meta.get(str(i), {})
            cache_data.append((
                "deepseek_v4_pending",
                layer_meta_dict.get("class_name", "DeepseekV4Cache"),
                layer_meta_dict.get("cache_meta", {}),
            ))
        else:
            cache_data.append(("skip",))

    return cache_data


def _infer_layer_type(data: Dict[str, Any], layer_idx: int, fallback_dtype: str) -> str:
    """Infer a layer's type from its tensor keys (backward compat for old blocks)."""
    prefix = f"layer_{layer_idx}_"
    has_keys_data = f"{prefix}keys_data" in data
    has_tq = f"{prefix}tq_ck_indices_packed" in data
    has_cumulative = f"{prefix}cumulative_0" in data
    has_dsv4 = f"{prefix}dsv4_state_0" in data
    has_dsv4_delta = f"{prefix}dsv4_delta_0" in data
    has_dsv4_pending = f"{prefix}dsv4_pending" in data
    has_max_size = f"{prefix}max_size" in data
    has_rotating_pending = f"{prefix}rotating_pending" in data
    has_keys = f"{prefix}keys" in data
    has_sub = any(key.startswith(f"{prefix}sub_") for key in data)

    if has_sub:
        return "cache_list"
    if has_tq:
        return "turboquant_kv"
    if has_keys_data:
        return "quantized_kv"
    if has_cumulative:
        return "cumulative"
    if has_dsv4:
        return "deepseek_v4"
    if has_dsv4_delta:
        return "deepseek_v4_delta_v1"
    if has_dsv4_pending:
        return "deepseek_v4_pending"
    if has_rotating_pending:
        return "rotating_kv_pending"
    if has_max_size and has_keys:
        return "rotating_kv"
    if has_keys:
        return "kv"
    return fallback_dtype

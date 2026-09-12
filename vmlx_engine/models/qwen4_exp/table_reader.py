"""Row-addressable safetensors reader for Qwen4-Exp's quantized PLE table."""

from __future__ import annotations

import fnmatch
import json
import logging
import mmap
import os
import struct
import threading
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from pathlib import Path

import mlx.core as mx
import numpy as np

from vmlx_engine.utils.jang_affine_storage import expand_packed_1bit_to_2bit_mlx

_DTYPES = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "U32": np.uint32,
    "I64": np.int64,
}

_MLX_DTYPES = {
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
    "U32": mx.uint32,
    "I64": mx.int64,
}

_PARALLEL_READ_MAX_ROWS = 128
_PARALLEL_READ_MAX_WORKERS = 16
_PREFETCH_MAX_ROWS = 8192
_PREFETCH_MAX_PACKED_BYTES = 8 * 1024 * 1024
logger = logging.getLogger(__name__)


def _parallel_ple_read_requested() -> bool:
    value = os.environ.get("VMLX_QWEN4_PLE_PARALLEL_READ", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def _host_ple_gather_requested() -> bool:
    # Qualification switch: host assembly changes dispatch, never the bundle.
    value = os.environ.get("VMLX_QWEN4_PLE_HOST_GATHER", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


class _PLEReadTicket:
    """Single-use, table-owned host buffers; never holds a decoded MLX array."""

    def __init__(self, owner, rows, future):
        self.owner = owner
        self.rows = rows
        self.future = future

    def close(self):
        self.owner._finish_prefetch(self, consumed=False)


class _SharedPreadFile:
    """One lazily opened descriptor shared by all tensors in a shard file."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None
        self._lock = threading.Lock()

    def _fileno(self) -> int:
        if self._fd is None:
            with self._lock:
                if self._fd is None:
                    self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def read(self, size: int, offset: int) -> bytes:
        data = os.pread(self._fileno(), size, offset)
        if len(data) != size:
            raise OSError(
                f"short PLE pread from {self.path}: {len(data)} != {size}"
            )
        return data

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


def _advise_random_access(array: np.memmap) -> bool:
    """Tell the kernel that PLE table pages are sparse random reads.

    The hash table lookup touches nearly unique rows spread across a very large
    file.  Default sequential readahead therefore fetches pages that the
    request will not use.  Keep this best-effort so platforms without
    ``madvise`` retain the original reader behavior.
    """
    mapping = getattr(array, "_mmap", None)
    advice = getattr(mmap, "MADV_RANDOM", None)
    madvise = getattr(mapping, "madvise", None)
    if advice is None or not callable(madvise):
        return False
    try:
        madvise(advice)
    except (OSError, TypeError, ValueError):
        return False
    return True


def resolve_jang_bit_map_spec(
    module_path: str,
    bit_map: Mapping[str, object],
) -> dict:
    """Resolve one converter-style wildcard/prefix quantization rule.

    The Qwen4-Exp converter matches the original tensor name (including its
    ``.weight`` suffix), while the MLX quantizer and PLE reader operate on
    module paths.  Accept both spellings plus the runtime's narrow wrapper
    aliases, choose the longest matching pattern, and reject equally-specific
    conflicting rules instead of depending on JSON insertion order.
    """
    if not isinstance(bit_map, Mapping):
        raise ValueError("qwen4_exp JANG bit_map must be an object")
    default = bit_map.get("default")
    if not isinstance(default, Mapping):
        raise ValueError("qwen4_exp JANG bit_map requires an object default")

    aliases = [
        str(module_path),
        str(module_path).replace("language_model.model", "language_model", 1),
        str(module_path).replace("language_model.mtp", "mtp", 1),
        str(module_path).replace("language_model.lm_head", "lm_head", 1),
    ]
    aliases.extend(f"{alias}.weight" for alias in tuple(aliases))
    aliases = list(dict.fromkeys(aliases))

    matches: list[tuple[int, str, Mapping[str, object]]] = []
    for raw_pattern, raw_spec in bit_map.items():
        pattern = str(raw_pattern)
        if pattern == "default":
            continue
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"qwen4_exp JANG bit_map rule {pattern!r} must be an object"
            )
        if any(
            fnmatch.fnmatch(alias, pattern)
            or alias.startswith(pattern)
            or fnmatch.fnmatch(alias, pattern + "*")
            for alias in aliases
        ):
            matches.append((len(pattern), pattern, raw_spec))

    selected: Mapping[str, object] = default
    if matches:
        best_length = max(length for length, _pattern, _spec in matches)
        best = [item for item in matches if item[0] == best_length]
        selected = best[0][2]
        conflicts = [pattern for _length, pattern, spec in best[1:] if spec != selected]
        if conflicts:
            raise ValueError(
                "conflicting equally-specific qwen4_exp JANG bit_map rules for "
                f"{module_path}: {best[0][1]}, " + ", ".join(conflicts)
            )

    spec = dict(selected)
    try:
        bits = int(spec["bits"])
        group_size = int(spec["group_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid qwen4_exp JANG quantization rule for {module_path}"
        ) from exc
    if bits not in {1, 2, 3, 4, 5, 6, 8} or group_size <= 0:
        raise ValueError(
            f"invalid qwen4_exp JANG quantization rule for {module_path}: "
            f"bits={bits}, group_size={group_size}"
        )
    spec.update({"bits": bits, "group_size": group_size})
    spec.setdefault("mode", "affine")
    return spec


def _module_aliases(module_path: str) -> tuple[str, ...]:
    """Return stable official/runtime aliases without set-order ambiguity."""
    candidates = [
        module_path,
        module_path.replace("model.language_model", "language_model.model", 1),
        module_path.replace("language_model.model", "model.language_model", 1),
        module_path.replace(".ple.ple_embedding.", ".ple."),
    ]
    candidates.extend(
        candidate.replace(".ple.ple_embedding.", ".ple.")
        for candidate in tuple(candidates)
    )
    return tuple(dict.fromkeys(candidates))


def _unique_mapping_override(
    mapping: dict,
    aliases: tuple[str, ...],
    *,
    label: str,
) -> dict | None:
    """Resolve one semantic module override and reject conflicting aliases."""
    matches = [
        (alias, mapping[alias])
        for alias in aliases
        if isinstance(mapping.get(alias), dict)
    ]
    if not matches:
        return None
    first_alias, first_value = matches[0]
    conflicts = [alias for alias, value in matches[1:] if value != first_value]
    if conflicts:
        raise ValueError(
            f"conflicting {label} aliases for {first_alias}: " + ", ".join(conflicts)
        )
    return dict(first_value)


def _validate_affine_layout(
    *,
    weight_shape: tuple[int, ...],
    weight_dtype: str,
    scales_shape: tuple[int, ...],
    biases_shape: tuple[int, ...],
    group_size: int,
    logical_bits: int,
    storage_bits: int,
    head_dim: int,
) -> None:
    if storage_bits not in {1, 2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported packed PLE storage_bits={storage_bits}")
    if logical_bits not in {1, 2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported PLE logical bits={logical_bits}")
    if storage_bits != 1 and storage_bits != logical_bits:
        raise ValueError(
            "PLE storage bits must equal logical bits unless using the lossless "
            f"affine-1 expansion: storage={storage_bits}, logical={logical_bits}"
        )
    if group_size <= 0 or head_dim <= 0:
        raise ValueError("PLE group size and head dimension must be positive")
    if weight_dtype != "U32":
        raise ValueError(f"PLE packed weight must be U32, got {weight_dtype}")
    if scales_shape != biases_shape:
        raise ValueError(
            "PLE scales and biases must have identical shapes: "
            f"{scales_shape} != {biases_shape}"
        )
    if len(weight_shape) != 2 or len(scales_shape) != 2:
        raise ValueError("PLE weight/scales/biases must all be rank 2")
    if not (weight_shape[0] == scales_shape[0] == biases_shape[0]):
        raise ValueError("PLE weight/scales/biases row counts differ")

    runtime_bits = 2 if storage_bits == 1 else logical_bits
    expected_storage_cols = (head_dim * storage_bits + 31) // 32
    expected_runtime_cols = (head_dim * runtime_bits + 31) // 32
    expanded_cols = weight_shape[1] * (2 if storage_bits == 1 else 1)
    expected_scale_cols = (head_dim + group_size - 1) // group_size
    if weight_shape[1] != expected_storage_cols:
        raise ValueError(
            "PLE packed width does not match the configured head dimension: "
            f"got={weight_shape[1]}, expected={expected_storage_cols}"
        )
    if expanded_cols != expected_runtime_cols:
        raise ValueError(
            "PLE runtime packed width does not match affine expansion: "
            f"got={expanded_cols}, expected={expected_runtime_cols}"
        )
    if scales_shape[1] != expected_scale_cols:
        raise ValueError(
            "PLE scale width does not match the configured head dimension: "
            f"got={scales_shape[1]}, expected={expected_scale_cols}"
        )


class SafetensorsRowReader:
    """Memory-map one two-dimensional safetensors tensor and fetch rows only."""

    def __init__(
        self,
        path: str | Path,
        tensor_name: str,
        *,
        pread_file: _SharedPreadFile | None = None,
    ):
        path = Path(path)
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
        info = header[tensor_name]
        self.dtype_tag = info["dtype"]
        if self.dtype_tag not in _DTYPES:
            raise ValueError(f"unsupported PLE tensor dtype {self.dtype_tag!r}")
        self.shape = tuple(info["shape"])
        if len(self.shape) != 2:
            raise ValueError(f"PLE tensor must be rank 2, got {self.shape}")
        start, _end = info["data_offsets"]
        self.data_offset = 8 + header_len + start
        self.mm = np.memmap(
            path,
            dtype=_DTYPES[self.dtype_tag],
            mode="r",
            offset=self.data_offset,
            shape=self.shape,
        )
        self.row_bytes = int(np.dtype(_DTYPES[self.dtype_tag]).itemsize)
        self.row_bytes *= int(np.prod(self.shape[1:], dtype=np.int64))
        self._pread_file = pread_file
        self.random_access_advised = _advise_random_access(self.mm)

    def rows(
        self, indices: np.ndarray, *, preserve_bits: bool = False
    ) -> np.ndarray:
        """Gather rows from the mmap; each distinct row is copied once.

        A PLE gather repeats rows whenever an n-gram recurs inside the chunk
        (common at prefill: heads x tokens); the pread path already reads
        unique rows, this path copied every duplicate."""
        indices = np.asarray(indices)
        flat = indices.reshape(-1)
        if flat.size > 1:
            unique, inverse = np.unique(flat, return_inverse=True)
            if unique.size < flat.size:
                values = np.asarray(self.mm[unique])[inverse].reshape(
                    *indices.shape, *self.shape[1:]
                )
            else:
                values = self.mm[indices]
        else:
            values = self.mm[indices]
        if self.dtype_tag == "BF16" and not preserve_bits:
            raw = np.asarray(values, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(values)

    def rows_pread(
        self, indices: np.ndarray, *, preserve_bits: bool = False
    ) -> np.ndarray:
        """Read selected rows without faulting the process-wide mmap."""

        if self._pread_file is None:
            raise RuntimeError("PLE pread source is unavailable")
        indices = np.asarray(indices, dtype=np.int64)
        flat = indices.reshape(-1)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.shape[0]):
            raise IndexError("PLE row is outside the tensor")
        unique, inverse = np.unique(flat, return_inverse=True)
        host = np.empty((unique.size, *self.shape[1:]), dtype=_DTYPES[self.dtype_tag])
        for position, row in enumerate(unique):
            raw = self._pread_file.read(
                self.row_bytes,
                self.data_offset + int(row) * self.row_bytes,
            )
            host[position] = np.frombuffer(
                raw,
                dtype=_DTYPES[self.dtype_tag],
                count=int(np.prod(self.shape[1:], dtype=np.int64)),
            ).reshape(self.shape[1:])
        values = host[inverse].reshape(*indices.shape, *self.shape[1:])
        if self.dtype_tag == "BF16" and not preserve_bits:
            raw = np.asarray(values, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return values

    @property
    def mlx_dtype(self):
        return _MLX_DTYPES[self.dtype_tag]

    def mlx_rows(self, indices: np.ndarray) -> mx.array:
        """Return rows with the safetensors dtype preserved for MLX kernels."""
        values = self.rows(indices)
        return mx.array(values).astype(self.mlx_dtype)


class _AffineShard:
    def __init__(
        self,
        model_dir: Path,
        weight_map: dict[str, str],
        module_path: str,
        quant_spec: dict,
        storage_bits: int | None,
        expected_head_dim: int,
        pread_files: dict[Path, _SharedPreadFile] | None = None,
    ):
        def reader(suffix: str) -> SafetensorsRowReader:
            key = f"{module_path}.{suffix}"
            path = model_dir / weight_map[key]
            pread_file = None
            if pread_files is not None:
                pread_file = pread_files.get(path)
                if pread_file is None:
                    pread_file = _SharedPreadFile(path)
                    pread_files[path] = pread_file
            return SafetensorsRowReader(path, key, pread_file=pread_file)

        self.weight = reader("weight")
        self.scales = reader("scales")
        self.biases = reader("biases")
        self.group_size = int(quant_spec["group_size"])
        self.logical_bits = int(quant_spec["bits"])
        self.storage_bits = int(storage_bits or self.logical_bits)
        self.mode = str(quant_spec.get("mode", "affine"))
        self.head_dim = int(expected_head_dim)
        if self.mode != "affine":
            raise ValueError(f"PLE row reader requires affine mode, got {self.mode!r}")
        if self.scales.dtype_tag != self.biases.dtype_tag:
            raise ValueError(
                "PLE scales and biases must have identical dtypes: "
                f"{self.scales.dtype_tag} != {self.biases.dtype_tag}"
            )
        if self.scales.dtype_tag not in {"BF16", "F16", "F32"}:
            raise ValueError(
                f"PLE affine parameters require a floating dtype, got {self.scales.dtype_tag}"
            )
        _validate_affine_layout(
            weight_shape=self.weight.shape,
            weight_dtype=self.weight.dtype_tag,
            scales_shape=self.scales.shape,
            biases_shape=self.biases.shape,
            group_size=self.group_size,
            logical_bits=self.logical_bits,
            storage_bits=self.storage_bits,
            head_dim=self.head_dim,
        )

    @property
    def rows_count(self) -> int:
        return self.weight.shape[0]

    @property
    def output_dtype(self):
        return self.scales.mlx_dtype

    @property
    def layout_signature(self) -> tuple:
        """Only rows with identical dequantization contracts may share a call.

        Artifact/row identity stays in the selection, not this grouping key:
        the arrays carry each shard's actual weights, scales and biases.
        There is no cross-table or persistent decoded-row cache here.
        """
        return (
            self.logical_bits, self.storage_bits, self.group_size, self.mode,
            self.head_dim,
            self.weight.dtype_tag, self.weight.shape[1:],
            self.scales.dtype_tag, self.scales.shape[1:],
            self.biases.dtype_tag, self.biases.shape[1:],
        )

    def gather_mlx(
        self,
        rows: np.ndarray,
        profile: dict[str, float] | None = None,
    ) -> mx.array:
        if profile is None:
            packed = self.weight.mlx_rows(rows)
            scales = self.scales.mlx_rows(rows)
            biases = self.biases.mlx_rows(rows)
        else:
            started = time.perf_counter()
            host_rows = self.read_rows(rows)
            profile["ssd_rows_cpu_ms"] = profile.get("ssd_rows_cpu_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
            return self.dequantize_rows_mlx(host_rows, profile=profile)

        return self._dequantize_mlx(packed, scales, biases, profile=profile)

    def read_rows(
        self,
        rows: np.ndarray,
        *,
        use_pread: bool = False,
        preserve_bits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        read = "rows_pread" if use_pread else "rows"
        kwargs = {"preserve_bits": True} if preserve_bits else {}
        return tuple(
            getattr(reader, read)(rows, **kwargs)
            for reader in (self.weight, self.scales, self.biases)
        )

    def dequantize_rows_mlx(
        self,
        host_rows: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        profile: dict[str, float] | None = None,
        preserve_bits: bool = False,
    ) -> mx.array:
        started = time.perf_counter() if profile is not None else None
        def upload(host, reader):
            value = mx.array(host)
            if preserve_bits and reader.dtype_tag == "BF16":
                # uint16 -> BF16 bit view, not an integer-to-float conversion.
                return value.view(mx.bfloat16)
            return value.astype(reader.mlx_dtype)

        packed = upload(host_rows[0], self.weight)
        scales = upload(host_rows[1], self.scales)
        biases = upload(host_rows[2], self.biases)
        if profile is not None:
            mx.eval(packed, scales, biases)
            profile["host_to_mlx_ms"] = profile.get("host_to_mlx_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
        return self._dequantize_mlx(packed, scales, biases, profile=profile)

    def _dequantize_mlx(
        self,
        packed: mx.array,
        scales: mx.array,
        biases: mx.array,
        *,
        profile: dict[str, float] | None,
    ) -> mx.array:
        started = time.perf_counter() if profile is not None else None
        runtime_bits = self.logical_bits
        if self.storage_bits == 1:
            packed = expand_packed_1bit_to_2bit_mlx(packed)
            runtime_bits = 2
        values = mx.dequantize(
            packed,
            scales,
            biases,
            group_size=self.group_size,
            bits=runtime_bits,
            mode=self.mode,
            dtype=self.output_dtype,
        )
        if profile is not None:
            mx.eval(values)
            profile["dequant_gpu_ms"] = profile.get("dequant_gpu_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
        if values.shape[-1] != self.head_dim:
            raise ValueError(
                "PLE dequantized row width differs from the model contract: "
                f"got={values.shape[-1]}, expected={self.head_dim}"
            )
        return values

    def gather(self, rows: np.ndarray) -> np.ndarray:
        values = self.gather_mlx(rows)
        mx.eval(values)
        return np.asarray(values).astype(np.float32, copy=False)


class FileBackedQuantizedNGramTable:
    """Gather/dequantize only requested PLE rows from the checkpoint shards."""

    def __init__(
        self,
        model_dir: str | Path,
        module_key_format: str,
        n_shards: int,
        expected_head_dim: int,
        index_name: str = "model.safetensors.index.json",
        bit_map: Mapping[str, object] | None = None,
    ):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        weight_map = json.loads((model_dir / index_name).read_text())["weight_map"]
        quantization = config.get("quantization") or {}
        default_spec = {
            "bits": quantization.get("bits", 4),
            "group_size": quantization.get("group_size", 64),
            "mode": quantization.get("mode", "affine"),
        }

        storage_manifest = {}
        jang_path = model_dir / "jang_config.json"
        jang = None
        if jang_path.is_file():
            jang = json.loads(jang_path.read_text())
        elif isinstance(config.get("jang_config"), dict):
            jang = config["jang_config"]
        elif isinstance(config.get("jang"), dict):
            jang = config["jang"]
        if isinstance(jang, dict):
            jq = jang.get("quantization") or {}
            storage_manifest = jq.get("tensor_quantization_manifest") or {}

        if n_shards <= 0:
            raise ValueError("PLE n_shards must be positive")
        self.shards = []
        self._pread_files: dict[Path, _SharedPreadFile] = {}
        for shard_index in range(n_shards):
            module_path = module_key_format.format(shard_index)
            spec = (
                resolve_jang_bit_map_spec(module_path, bit_map)
                if bit_map is not None
                else dict(default_spec)
            )
            aliases = _module_aliases(module_path)
            override = _unique_mapping_override(
                quantization,
                aliases,
                label="PLE quantization",
            )
            if isinstance(override, dict):
                spec.update(override)
            manifest_spec = _unique_mapping_override(
                storage_manifest,
                aliases,
                label="PLE storage manifest",
            )
            storage_bits = None
            if isinstance(manifest_spec, dict):
                storage_bits = manifest_spec.get(
                    "storage_bits", manifest_spec.get("bits")
                )
            self.shards.append(
                _AffineShard(
                    model_dir,
                    weight_map,
                    module_path,
                    spec,
                    storage_bits,
                    expected_head_dim,
                    self._pread_files,
                )
            )
        self.per = self.shards[0].rows_count
        self.head_dim = int(expected_head_dim)
        self.output_dtype = self.shards[0].output_dtype
        if self.per <= 0:
            raise ValueError("PLE shard 0 is empty")
        for shard_index, shard in enumerate(self.shards):
            if shard.head_dim != self.head_dim:
                raise ValueError(f"PLE shard {shard_index} head dimension differs")
            if shard.output_dtype != self.output_dtype:
                raise ValueError(f"PLE shard {shard_index} output dtype differs")
            if shard_index < len(self.shards) - 1 and shard.rows_count != self.per:
                raise ValueError(
                    f"PLE shard {shard_index} has {shard.rows_count} rows; "
                    f"expected {self.per}"
                )
            if shard_index == len(self.shards) - 1 and not (
                0 < shard.rows_count <= self.per
            ):
                raise ValueError("PLE final shard row count is invalid")
        self.total_rows = sum(shard.rows_count for shard in self.shards)
        readers = tuple(
            reader
            for shard in self.shards
            for reader in (shard.weight, shard.scales, shard.biases)
        )
        self.random_access_reader_count = len(readers)
        self.random_access_advised_readers = sum(
            bool(reader.random_access_advised) for reader in readers
        )
        self._parallel_read = _parallel_ple_read_requested()
        self._host_assembly = _host_ple_gather_requested()
        self._prefetch_enabled = os.environ.get("VMLX_QWEN4_PLE_PREFETCH") == "1"
        self._prefetch_lock = threading.Lock()
        self._prefetch_pool = None
        self._prefetch_ticket = None
        self._closed = False
        self.prefetch_stats = {"submitted": 0, "consumed": 0, "discarded": 0,
                               "capacity_fallbacks": 0}
        self.host_gather_stats = {
            "calls": 0, "rows": 0, "unique_rows": 0,
            "shards": 0, "layout_groups": 0,
        }
        self._read_pool = (
            ThreadPoolExecutor(
                max_workers=min(_PARALLEL_READ_MAX_WORKERS, n_shards),
                thread_name_prefix="vmlx-ple-read",
            )
            if self._parallel_read
            else None
        )

    def close(self) -> None:
        # The outer reader may be awaiting shard workers. Drain it before
        # shutting down that distinct pool or closing any shared descriptors.
        lock = getattr(self, "_prefetch_lock", None)
        if lock is not None:
            with lock:
                self._closed = True
                prefetch_pool = self._prefetch_pool
                ticket = self._prefetch_ticket
            if prefetch_pool is not None:
                prefetch_pool.shutdown(wait=True, cancel_futures=True)
            if ticket is not None:
                ticket.close()
            self._prefetch_pool = None
        pool = getattr(self, "_read_pool", None)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            self._read_pool = None
        for pread_file in getattr(self, "_pread_files", {}).values():
            pread_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def gather(self, flat_rows: np.ndarray) -> np.ndarray:
        values = self.gather_mlx(flat_rows)
        mx.eval(values)
        return np.asarray(values).astype(np.float32, copy=False)

    def gather_mlx(
        self,
        flat_rows: np.ndarray,
        profile: dict[str, float] | None = None,
        *,
        prepared: _PLEReadTicket | None = None,
    ) -> mx.array:
        """Gather random SSD rows and keep dequantized values on the MLX path."""
        flat_rows = np.asarray(flat_rows, dtype=np.int64).reshape(-1)
        if flat_rows.size and int(flat_rows.min()) < 0:
            raise IndexError("PLE row must be non-negative")
        if flat_rows.size and int(flat_rows.max()) >= self.total_rows:
            raise IndexError("PLE row exceeds the configured n-gram table")
        if getattr(self, "_closed", False):
            raise RuntimeError("PLE table is closed")
        if prepared is not None:
            if prepared.owner is not self:
                raise ValueError("PLE read ticket belongs to a different table")
            consumed = False
            try:
                future = prepared.future
                if future is None:
                    raise ValueError("PLE read ticket was already released")
                if not np.array_equal(flat_rows, prepared.rows):
                    raise ValueError("PLE read ticket does not match the exact row IDs")
                hosts = future.result()
                result = self._materialize_host_assembled(flat_rows.size, hosts, profile)
                consumed = True
                return result
            finally:
                self._finish_prefetch(prepared, consumed=consumed)
        if getattr(self, "_host_assembly", False):
            return self._gather_host_assembled(flat_rows, profile)
        shard_indices = flat_rows // self.per
        local_rows = flat_rows % self.per
        out = mx.zeros(
            (flat_rows.size, self.head_dim), dtype=self.output_dtype
        )
        unique_shards = np.unique(shard_indices)
        selections = [
            (
                int(shard_index),
                np.nonzero(shard_indices == shard_index)[0],
            )
            for shard_index in unique_shards
        ]
        pool = getattr(self, "_read_pool", None)
        parallel = (
            pool is not None
            and 1 < len(selections)
            and flat_rows.size <= _PARALLEL_READ_MAX_ROWS
            and all(
                hasattr(self.shards[shard_index], "read_rows")
                for shard_index, _selected in selections
            )
        )
        futures = {}
        host_batches = {}
        if parallel:
            if not getattr(self, "_parallel_read_logged", False):
                logger.info(
                    "Qwen PLE parallel pread active: selected_shards=%d rows=%d",
                    len(selections), flat_rows.size,
                )
                self._parallel_read_logged = True
            started = time.perf_counter() if profile is not None else None
            futures = {
                shard_index: pool.submit(
                    self.shards[shard_index].read_rows,
                    local_rows[selected],
                    use_pread=True,
                )
                for shard_index, selected in selections
            }
            # A failed shard must not leave sibling reads running past the
            # caller's error cleanup / descriptor lifetime.
            wait(futures.values())
            host_batches = {
                shard_index: future.result()
                for shard_index, future in futures.items()
            }
            if profile is not None:
                read_wall_ms = (time.perf_counter() - started) * 1000.0
                # Preserve the established aggregate key for profile consumers
                # while exposing the parallel wall-clock component separately.
                profile["ssd_rows_cpu_ms"] = profile.get(
                    "ssd_rows_cpu_ms", 0.0
                ) + read_wall_ms
                profile["ssd_rows_parallel_wall_ms"] = profile.get(
                    "ssd_rows_parallel_wall_ms", 0.0
                ) + read_wall_ms

        for shard_index, selected in selections:
            selected_mx = mx.array(selected.astype(np.uint32))
            shard = self.shards[shard_index]
            if parallel:
                values = shard.dequantize_rows_mlx(
                    host_batches[shard_index],
                    profile=profile,
                )
            else:
                values = shard.gather_mlx(local_rows[selected], profile=profile)
            out[selected_mx] = values
        if profile is not None:
            started = time.perf_counter()
            mx.eval(out)
            profile["scatter_gpu_ms"] = profile.get("scatter_gpu_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
        return out

    def _gather_host_assembled(
        self, flat_rows: np.ndarray, profile: dict[str, float] | None
    ) -> mx.array:
        """Read on the host, upload/dequantize once per *actual* layout.

        Deduplication applies to row IDs within this call only. It does not
        reuse rows from a previous history/model or retain an unbounded table.
        All MLX work stays on the caller's generation stream; workers do I/O.
        """
        if not flat_rows.size:
            return mx.zeros((0, self.head_dim), dtype=self.output_dtype)
        hosts = self._read_host_assembled(flat_rows, profile=profile)
        return self._materialize_host_assembled(flat_rows.size, hosts, profile)

    def prefetch_rows(self, flat_rows: np.ndarray) -> _PLEReadTicket | None:
        """Prepare one bounded exact selection without running MLX on a worker.

        Callers must consume or close the ticket in a finally block. A busy or
        oversized request uses the unchanged synchronous path, not a queue.
        """
        if not self._prefetch_enabled or not self._host_assembly:
            return None
        rows = np.array(flat_rows, dtype=np.int64, copy=True).reshape(-1)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.total_rows):
            raise IndexError("PLE prefetch row exceeds the configured n-gram table")
        if not rows.size:
            return None
        if rows.size > _PREFETCH_MAX_ROWS:
            self.prefetch_stats["capacity_fallbacks"] += 1
            return None
        unique = np.unique(rows)
        shard_ids, counts = np.unique(unique // self.per, return_counts=True)
        packed_bytes = sum(
            int(count) * sum(reader.row_bytes for reader in
                             (self.shards[int(s)].weight, self.shards[int(s)].scales,
                              self.shards[int(s)].biases))
            for s, count in zip(shard_ids, counts)
        )
        if packed_bytes > _PREFETCH_MAX_PACKED_BYTES:
            self.prefetch_stats["capacity_fallbacks"] += 1
            return None
        rows.setflags(write=False)
        with self._prefetch_lock:
            if self._closed:
                raise RuntimeError("PLE table is closed")
            if self._prefetch_ticket is not None:
                self.prefetch_stats["capacity_fallbacks"] += 1
                return None
            if self._prefetch_pool is None:
                self._prefetch_pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="vmlx-ple-prefetch"
                )
            future = self._prefetch_pool.submit(self._read_host_assembled, rows,
                                                use_pread=True)
            ticket = _PLEReadTicket(self, rows, future)
            self._prefetch_ticket = ticket
            self.prefetch_stats["submitted"] += 1
        return ticket

    def _finish_prefetch(self, ticket: _PLEReadTicket, *, consumed: bool) -> None:
        if ticket.owner is not self:
            raise ValueError("PLE read ticket belongs to a different table")
        future = ticket.future
        if future is None:
            return
        # Drain even when the caller failed before reaching its PLE layer.
        # Do not mask that caller exception with an abandoned read exception.
        try:
            future.exception()
        except CancelledError:
            pass
        with self._prefetch_lock:
            if ticket.future is None:
                return
            ticket.future = None
            if self._prefetch_ticket is ticket:
                self._prefetch_ticket = None
            key = "consumed" if consumed else "discarded"
            if consumed and not self.prefetch_stats["consumed"]:
                logger.info("Qwen PLE host prefetch consumed: rows=%d "
                            "max_rows=%d max_packed_bytes=%d stream=caller",
                            ticket.rows.size, _PREFETCH_MAX_ROWS,
                            _PREFETCH_MAX_PACKED_BYTES)
            self.prefetch_stats[key] += 1

    def _read_host_assembled(self, flat_rows: np.ndarray, *,
                             profile=None, use_pread: bool = False):
        """Host-only preparation, shared by sync and bounded prefetch paths."""
        started = time.perf_counter() if profile is not None else None
        unique_rows, inverse = np.unique(flat_rows, return_inverse=True)
        shard_ids = unique_rows // self.per
        local_rows = unique_rows % self.per
        selections = [
            (int(shard_id), np.flatnonzero(shard_ids == shard_id))
            for shard_id in np.unique(shard_ids)
        ]
        groups = {}
        for shard_id, selected in selections:
            signature = self.shards[shard_id].layout_signature
            groups.setdefault(signature, []).append((shard_id, selected))

        pool = getattr(self, "_read_pool", None)
        parallel = (
            pool is not None and len(selections) > 1
            and unique_rows.size <= _PARALLEL_READ_MAX_ROWS
        )
        host_batches = {}
        if parallel:
            futures = {
                shard_id: pool.submit(
                    self.shards[shard_id].read_rows, local_rows[selected],
                    use_pread=True, preserve_bits=True,
                )
                for shard_id, selected in selections
            }
            wait(futures.values())
            host_batches = {
                shard_id: future.result() for shard_id, future in futures.items()
            }
        else:
            host_batches = {
                shard_id: self.shards[shard_id].read_rows(
                    local_rows[selected], use_pread=use_pread, preserve_bits=True
                )
                for shard_id, selected in selections
            }
        if profile is not None:
            profile["ssd_rows_cpu_ms"] = profile.get("ssd_rows_cpu_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0

        assembled = []
        for members in groups.values():
            shard = self.shards[members[0][0]]
            selected = np.concatenate([indices for _, indices in members])
            hosts = tuple(
                np.concatenate([host_batches[shard_id][i] for shard_id, _ in members])
                for i in range(3)
            )
            assembled.append((shard, selected, hosts))
        return inverse, unique_rows.size, len(selections), assembled

    def _materialize_host_assembled(self, row_count, prepared, profile):
        """Only the caller uploads, dequantizes and restores original row order."""
        inverse, unique_count, shard_count, groups = prepared
        out = None
        for shard, selected, hosts in groups:
            values = shard.dequantize_rows_mlx(
                hosts, profile=profile, preserve_bits=True
            )
            if len(groups) == 1:
                # unique_rows is sorted, hence members are already in order.
                out = values
            else:
                if out is None:
                    out = mx.zeros(
                        (unique_count, self.head_dim), dtype=self.output_dtype
                    )
                out[mx.array(selected.astype(np.uint32))] = values
        out = out[mx.array(inverse.astype(np.uint32))]
        if profile is not None:
            started = time.perf_counter()
            mx.eval(out)
            profile["scatter_gpu_ms"] = profile.get("scatter_gpu_ms", 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
        stats = self.host_gather_stats
        if not stats["calls"]:
            logger.info(
                "Qwen PLE host assembly active: rows=%d unique_rows=%d "
                "selected_shards=%d layout_groups=%d",
                row_count, unique_count, shard_count, len(groups),
            )
        stats["calls"] += 1
        stats["rows"] += int(row_count)
        stats["unique_rows"] += int(unique_count)
        stats["shards"] += shard_count
        stats["layout_groups"] += len(groups)
        return out

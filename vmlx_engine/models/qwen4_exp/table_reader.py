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
from concurrent.futures import ThreadPoolExecutor
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
logger = logging.getLogger(__name__)


def _parallel_ple_read_requested() -> bool:
    value = os.environ.get("VMLX_QWEN4_PLE_PARALLEL_READ", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


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

    def rows(self, indices: np.ndarray) -> np.ndarray:
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
        if self.dtype_tag == "BF16":
            raw = np.asarray(values, dtype=np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(values)

    def rows_pread(self, indices: np.ndarray) -> np.ndarray:
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
        if self.dtype_tag == "BF16":
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        read = "rows_pread" if use_pread else "rows"
        return (
            getattr(self.weight, read)(rows),
            getattr(self.scales, read)(rows),
            getattr(self.biases, read)(rows),
        )

    def dequantize_rows_mlx(
        self,
        host_rows: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        profile: dict[str, float] | None = None,
    ) -> mx.array:
        started = time.perf_counter() if profile is not None else None
        packed = mx.array(host_rows[0]).astype(self.weight.mlx_dtype)
        scales = mx.array(host_rows[1]).astype(self.scales.mlx_dtype)
        biases = mx.array(host_rows[2]).astype(self.biases.mlx_dtype)
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
    """Gather/dequantize only requested PLE rows from its 128 SSD shards."""

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
        self._read_pool = (
            ThreadPoolExecutor(
                max_workers=min(_PARALLEL_READ_MAX_WORKERS, n_shards),
                thread_name_prefix="vmlx-ple-read",
            )
            if self._parallel_read
            else None
        )

    def close(self) -> None:
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
    ) -> mx.array:
        """Gather random SSD rows and keep dequantized values on the MLX path."""
        flat_rows = np.asarray(flat_rows, dtype=np.int64).reshape(-1)
        if flat_rows.size and int(flat_rows.min()) < 0:
            raise IndexError("PLE row must be non-negative")
        if flat_rows.size and int(flat_rows.max()) >= self.total_rows:
            raise IndexError("PLE row exceeds the configured n-gram table")
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

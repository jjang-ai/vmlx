# SPDX-License-Identifier: Apache-2.0
"""Flash MoE SSD expert streaming for vMLX.

Streams MoE expert weights from SSD on-demand, enabling massive MoE models
(35B-397B parameters) to run on Macs with limited RAM by keeping only active
experts cached in a slot-bank LRU cache.

Reuses ExpertIndex from smelt_loader.py for expert tensor location scanning.

Key components:
    SlotBankCache — Thread-safe LRU cache for expert weight sets
    FlashMoEExpertLoader — On-demand expert loading from safetensors via pread

Reference: ANeMLL Flash-MoE (Anemll/anemll-flash-mlx)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .smelt_loader import (
    ExpertIndex,
    LayerExpertInfo,
    ProjectionTensors,
    TensorInfo,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Safetensors dtype mapping (shared with smelt_loader)
# ═══════════════════════════════════════════════════════════════════════════════

_NP_DTYPE_MAP = {
    "U32": np.uint32,
    "F16": np.float16,
    "BF16": np.float16,  # loaded as float16, MLX handles bfloat16 conversion
    "F32": np.float32,
    "I32": np.int32,
    "U8": np.uint8,
}

# MLX dtype mapping for array creation
_MLX_DTYPE_MAP = {
    "U32": mx.uint32,
    "F16": mx.float16,
    "BF16": mx.bfloat16,
    "F32": mx.float32,
    "I32": mx.int32,
    "U8": mx.uint8,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertWeightSet — all tensors for one expert in one layer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExpertWeightSet:
    """All weight tensors for a single expert in a single MoE layer.

    For a 3-projection MoE (Qwen/Mistral/Gemma/MiniMax), each projection
    has up to 3 tensors: weight (uint32 packed), scales (float16), biases.
    Total: up to 9 tensors per expert.

    For Nemotron 2-projection MoE: up to 6 tensors per expert.
    """

    layer_idx: int
    expert_idx: int
    # Projection name → tensor name → MLX array
    # e.g. {"gate_proj": {"weight": mx.array, "scales": mx.array}, ...}
    tensors: Dict[str, Dict[str, mx.array]] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        total = 0
        for proj_tensors in self.tensors.values():
            for arr in proj_tensors.values():
                total += arr.nbytes
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# SlotBankCache — LRU cache for expert weight sets
# ═══════════════════════════════════════════════════════════════════════════════


class SlotBankCache:
    """Thread-safe LRU cache for MoE expert weight sets.

    Each slot holds one ExpertWeightSet (all tensors for one expert in one
    layer). When the cache is full, the least-recently-used slot is evicted.

    Args:
        max_slots: Maximum number of expert weight sets to cache.
    """

    def __init__(self, max_slots: int = 64):
        self._max_slots = max_slots
        self._cache: OrderedDict[str, ExpertWeightSet] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(layer_idx: int, expert_idx: int) -> str:
        return f"{layer_idx}:{expert_idx}"

    def get(self, layer_idx: int, expert_idx: int) -> Optional[ExpertWeightSet]:
        """Retrieve expert from cache, promoting to most-recently-used."""
        key = self._key(layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, expert: ExpertWeightSet) -> None:
        """Insert expert into cache, evicting LRU if full."""
        key = self._key(expert.layer_idx, expert.expert_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = expert
                return
            if len(self._cache) >= self._max_slots:
                self._cache.popitem(last=False)  # evict LRU
            self._cache[key] = expert

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "slots_used": len(self._cache),
                "max_slots": self._max_slots,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Single-tensor pread loader
# ═══════════════════════════════════════════════════════════════════════════════


def _pread_tensor(ti: TensorInfo, expert_idx: int) -> mx.array:
    """Read a single expert's slice from a safetensors file using pread.

    Expert tensors in SwitchLinear are stored as [num_experts, ...] — this
    function reads only the bytes for expert_idx.

    Uses raw uint8 reads + view() (same approach as smelt_loader) to handle
    BF16 correctly — numpy doesn't have bfloat16, so we read raw bytes and
    let view(np.float16) reinterpret for 2-byte types.

    Args:
        ti: TensorInfo with file path, absolute offset, shape, dtype.
        expert_idx: Which expert to extract (index into dim 0).

    Returns:
        MLX array with shape = ti.shape[1:] (expert dim removed).
    """
    np_dtype = _NP_DTYPE_MAP.get(ti.dtype, np.uint8)
    bytes_per_element = np.dtype(np_dtype).itemsize

    if not ti.shape or len(ti.shape) < 2:
        # Scalar or 1D tensor — load entire tensor
        total_elements = 1
        for s in ti.shape:
            total_elements *= s
        num_bytes = total_elements * bytes_per_element
        raw = np.fromfile(
            str(ti.file_path), dtype=np.uint8, count=num_bytes,
            offset=ti.abs_offset,
        )
        return mx.array(raw.view(np_dtype).reshape(ti.shape))

    # Shape is [num_experts, *rest] — compute byte offset for expert_idx
    expert_shape = ti.shape[1:]
    expert_elements = 1
    for s in expert_shape:
        expert_elements *= s

    expert_bytes = expert_elements * bytes_per_element
    expert_offset = ti.abs_offset + expert_idx * expert_bytes

    # Read raw bytes then view as typed array (handles BF16 correctly)
    raw = np.fromfile(
        str(ti.file_path), dtype=np.uint8, count=expert_bytes,
        offset=expert_offset,
    )
    if raw.size != expert_bytes:
        raise RuntimeError(
            f"flash-moe: short read on expert {expert_idx} in "
            f"'{ti.file_path}' — expected {expert_bytes} bytes at "
            f"offset {expert_offset}, got {raw.size} bytes"
        )

    return mx.array(raw.view(np_dtype).reshape(expert_shape))


def _load_projection_expert(
    proj: ProjectionTensors, expert_idx: int
) -> Dict[str, mx.array]:
    """Load all tensors (weight, scales, biases) for one expert from one projection."""
    result = {}
    for suffix in ("weight", "scales", "biases"):
        ti = getattr(proj, suffix, None)
        if ti is not None:
            result[suffix] = _pread_tensor(ti, expert_idx)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FlashMoEExpertLoader — on-demand expert loading with slot-bank cache
# ═══════════════════════════════════════════════════════════════════════════════


class FlashMoEExpertLoader:
    """Loads MoE expert weights on-demand from SSD with slot-bank caching.

    Uses the ExpertIndex from smelt_loader to know WHERE each expert's
    tensors live in safetensors files (file path, byte offset, shape, dtype).
    Loads individual experts via numpy pread (no mmap, no full-file load).

    Thread pool handles parallel I/O for loading multiple experts at once
    (e.g., top-K experts per token all needed simultaneously).

    Args:
        expert_index: ExpertIndex mapping expert tensors to disk locations.
        cache: SlotBankCache for caching loaded expert weight sets.
        io_workers: Number of parallel I/O threads. Default 4.
    """

    def __init__(
        self,
        expert_index: ExpertIndex,
        cache: SlotBankCache,
        io_workers: int = 4,
    ):
        self._index = expert_index
        self._cache = cache
        self._io_workers = io_workers
        self._executor = ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="flash-moe-io"
        )
        self._load_count = 0
        self._load_time_ms = 0.0

    def load_expert(self, layer_idx: int, expert_idx: int) -> Optional[ExpertWeightSet]:
        """Load a single expert, checking cache first.

        Returns None if the expert is not in the index (layer has no MoE).
        """
        # Check cache first
        cached = self._cache.get(layer_idx, expert_idx)
        if cached is not None:
            return cached

        # Load from disk
        layer_info = self._index.layers.get(layer_idx)
        if layer_info is None:
            return None

        t0 = time.monotonic()
        expert = self._load_expert_from_disk(layer_info, expert_idx)
        elapsed_ms = (time.monotonic() - t0) * 1000

        self._load_count += 1
        self._load_time_ms += elapsed_ms

        # Cache for reuse
        self._cache.put(expert)
        return expert

    def load_experts_parallel(
        self, layer_idx: int, expert_indices: List[int]
    ) -> Dict[int, ExpertWeightSet]:
        """Load multiple experts for one layer in parallel.

        Checks cache first for each expert, only loads cache-misses from disk.
        Returns dict mapping expert_idx → ExpertWeightSet.
        """
        result: Dict[int, ExpertWeightSet] = {}
        to_load: List[int] = []

        # Check cache for all experts
        for eidx in expert_indices:
            cached = self._cache.get(layer_idx, eidx)
            if cached is not None:
                result[eidx] = cached
            else:
                to_load.append(eidx)

        if not to_load:
            return result

        # Parallel load from disk
        layer_info = self._index.layers.get(layer_idx)
        if layer_info is None:
            return result

        t0 = time.monotonic()

        futures = {
            eidx: self._executor.submit(
                self._load_expert_from_disk, layer_info, eidx
            )
            for eidx in to_load
        }

        for eidx, future in futures.items():
            try:
                expert = future.result()
                self._cache.put(expert)
                result[eidx] = expert
            except Exception as e:
                logger.warning(
                    "Flash MoE: failed to load expert %d in layer %d: %s",
                    eidx, layer_idx, e,
                )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._load_count += len(to_load)
        self._load_time_ms += elapsed_ms

        return result

    def _load_expert_from_disk(
        self, layer_info: LayerExpertInfo, expert_idx: int
    ) -> ExpertWeightSet:
        """Load all tensors for one expert from disk."""
        tensors: Dict[str, Dict[str, mx.array]] = {}

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj: Optional[ProjectionTensors] = getattr(layer_info, proj_name, None)
            if proj is None:
                continue
            proj_tensors = _load_projection_expert(proj, expert_idx)
            if proj_tensors:
                tensors[proj_name] = proj_tensors

        return ExpertWeightSet(
            layer_idx=layer_info.layer_idx,
            expert_idx=expert_idx,
            tensors=tensors,
        )

    def stats(self) -> dict:
        cache_stats = self._cache.stats()
        return {
            **cache_stats,
            "disk_loads": self._load_count,
            "disk_load_time_ms": round(self._load_time_ms, 2),
            "avg_load_ms": (
                round(self._load_time_ms / self._load_count, 2)
                if self._load_count > 0
                else 0.0
            ),
            "io_workers": self._io_workers,
        }

    def shutdown(self) -> None:
        """Shutdown the I/O thread pool."""
        self._executor.shutdown(wait=False)
        self._cache.clear()

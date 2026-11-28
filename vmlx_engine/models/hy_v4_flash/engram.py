# SPDX-License-Identifier: Apache-2.0
"""Disk-backed Engram N-gram memory tables & 4-stream iHC gating for Hy4-flash.

Design (zero wired GPU memory for 118.02 GiB of Engram tables):
- The 4 Engram embedding tables (layers 1, 11, 21, 31 — 1-based config IDs [2, 12, 22, 32])
  stay untouched on NVMe SSD inside their original safetensors shards
  (`model-00001..00004-of-00148.safetensors`).
- Each table is mapped read-only via `np.memmap(..., MADV_RANDOM)` directly at the tensor's
  exact byte offset (no copy, no conversion, 0 GiB GPU RAM!).
- On every token/chunk, 16 concurrent threads gather only the deduplicated hash-indexed
  BF16 (or FP8) rows (`16 rows x 96 dims = 3 KiB/token/layer`).
- A background 4-thread prefetch pool overlaps layer `11`'s SSD gather while GPU executes
  layers `1..10`, layer `21` during `11..20`, and layer `31` during `21..30` — completely
  hiding NVMe random-read latency behind transformer compute!
"""

from __future__ import annotations

import json
import mmap
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import HyV4FlashConfig

# Matches PyTorch nn.RMSNorm default eps = torch.finfo(torch.float32).eps
_ENGRAM_NORM_EPS = float(np.finfo(np.float32).eps)

_TAKE_LANES = 16
_TAKE_POOL = ThreadPoolExecutor(_TAKE_LANES, thread_name_prefix="hy4-engram-take")
_PREFETCH_POOL = ThreadPoolExecutor(4, thread_name_prefix="hy4-engram-prefetch")


def _read_safetensors_tensor_meta(path: Path, tensor_name: str) -> Tuple[int, Tuple[int, ...], str]:
    """Parse ONLY the tiny JSON header of a safetensors shard to locate a tensor's exact byte offset."""
    with path.open("rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
    entry = hdr[tensor_name]
    data_start = 8 + hdr_len + int(entry["data_offsets"][0])
    shape = tuple(int(x) for x in entry["shape"])
    dtype = str(entry["dtype"])
    return data_start, shape, dtype


class HyV4DiskEngramTable:
    """Zero-copy disk-backed Engram embedding table (`~29.5 GiB` BF16 per layer, `0 B` GPU RAM).

    Supports raw BF16 (`uint16` view -> `mx.array.view(mx.bfloat16)`) as well as FP8 E4M3
    if an Engram table is quantized in a future checkpoint.
    """

    def __init__(
        self,
        shard_path: Path,
        tensor_name: str,
        *,
        scale_tensor_name: Optional[str] = None,
    ) -> None:
        self.path = Path(shard_path)
        self.tensor_name = tensor_name
        offset, shape, dtype = _read_safetensors_tensor_meta(self.path, tensor_name)
        if len(shape) != 2:
            raise ValueError(f"Expected 2-D Engram table for {tensor_name}, got {shape}")
        self.num_rows, self.head_dim = int(shape[0]), int(shape[1])
        self.dtype_str = dtype
        if dtype == "BF16":
            self.rows = np.memmap(self.path, np.uint16, mode="r", offset=offset, shape=shape)
        elif dtype in ("F8_E4M3", "U8"):
            self.rows = np.memmap(self.path, np.uint8, mode="r", offset=offset, shape=shape)
        elif dtype == "F16":
            self.rows = np.memmap(self.path, np.float16, mode="r", offset=offset, shape=shape)
        elif dtype == "F32":
            self.rows = np.memmap(self.path, np.float32, mode="r", offset=offset, shape=shape)
        else:
            raise ValueError(f"Unsupported Engram table dtype {dtype!r} for {tensor_name}")

        self.scales: Optional[np.memmap] = None
        self.scale_block: int = self.head_dim
        if scale_tensor_name is not None:
            s_off, s_shape, s_dtype = _read_safetensors_tensor_meta(self.path, scale_tensor_name)
            s_np = np.uint8 if s_dtype in ("F8_E8M0", "U8") else (np.uint16 if s_dtype == "BF16" else np.float32)
            self.scales = np.memmap(self.path, s_np, mode="r", offset=s_off, shape=s_shape)
            self.scale_dtype_str = s_dtype
            self.scale_block = self.head_dim // int(s_shape[1]) if len(s_shape) == 2 and s_shape[1] > 0 else self.head_dim

        # Disable OS sequential read-ahead on random hash-indexed lookups (~1.5x faster cold NVMe rows/s)
        for mm in (self.rows, self.scales):
            if mm is not None:
                try:
                    mm._mmap.madvise(mmap.MADV_RANDOM)
                except (AttributeError, OSError, ValueError):
                    pass

    def _gather_unique(self, uniq: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n = int(uniq.shape[0])
        out_rows = np.empty((n, self.head_dim), dtype=self.rows.dtype)
        out_scales = (
            np.empty((n, self.scales.shape[1]), dtype=self.scales.dtype)
            if self.scales is not None
            else None
        )
        if n <= 32:
            np.take(self.rows, uniq, axis=0, out=out_rows)
            if self.scales is not None and out_scales is not None:
                np.take(self.scales, uniq, axis=0, out=out_scales)
            return out_rows, out_scales

        step = max(1, (n + _TAKE_LANES - 1) // _TAKE_LANES)

        def _take(lo: int) -> None:
            hi = min(n, lo + step)
            u = uniq[lo:hi]
            np.take(self.rows, u, axis=0, out=out_rows[lo:hi])
            if self.scales is not None and out_scales is not None:
                np.take(self.scales, u, axis=0, out=out_scales[lo:hi])

        list(_TAKE_POOL.map(_take, range(0, n, step)))
        return out_rows, out_scales

    def read(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Host-side thread-safe gather (no MLX calls): deduplicates ids and gathers from disk memmap."""
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        rr, ss = self._gather_unique(uniq)
        return inv.astype(np.int32, copy=False), rr, ss

    def prefetch(self, ids: np.ndarray, after: Optional[Future] = None) -> Future:
        """Schedule asynchronous disk gather on `_PREFETCH_POOL`, optionally chained after a prior future."""
        def _run():
            if after is not None:
                after.result()
            return self.read(ids)
        return _PREFETCH_POOL.submit(_run)

    def finish(self, shape: Tuple[int, ...], got: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]) -> mx.array:
        """Convert gathered host rows into BF16 MLX array of shape `(*shape, head_dim)`."""
        inv, rr, ss = got
        if self.dtype_str == "BF16":
            vals = mx.array(rr).view(mx.bfloat16)
        elif self.dtype_str in ("F8_E4M3", "U8"):
            vals_f = mx.from_fp8(mx.array(rr), dtype=mx.float32)
            if ss is not None:
                if self.scale_dtype_str in ("F8_E8M0", "U8"):
                    sc = (mx.array(ss).astype(mx.uint32) << 23).view(mx.float32)
                elif self.scale_dtype_str == "BF16":
                    sc = mx.array(ss).view(mx.bfloat16).astype(mx.float32)
                else:
                    sc = mx.array(ss, dtype=mx.float32)
                n = rr.shape[0]
                vals_f = (vals_f.reshape(n, -1, self.scale_block) * sc[..., None]).reshape(n, self.head_dim)
            vals = vals_f.astype(mx.bfloat16)
        elif self.dtype_str == "F16":
            vals = mx.array(rr).astype(mx.bfloat16)
        else:
            vals = mx.array(rr).astype(mx.bfloat16)
        return vals[mx.array(inv)].reshape(*shape, self.head_dim)

    def lookup(self, ids: np.ndarray) -> mx.array:
        return self.finish(tuple(ids.shape), self.read(ids))


class HyV4InMemoryEngramTable:
    """In-memory Engram table for unit tests and synthetic parity verification."""

    def __init__(self, weight: mx.array) -> None:
        self.weight = weight.astype(mx.bfloat16)
        self.num_rows, self.head_dim = int(weight.shape[0]), int(weight.shape[1])

    def prefetch(self, ids: np.ndarray, after: Optional[Future] = None):
        return ids

    def finish(self, shape: Tuple[int, ...], got) -> mx.array:
        idx = mx.array(np.asarray(got, dtype=np.int32).reshape(-1))
        return self.weight[idx].reshape(*shape, self.head_dim)

    def lookup(self, ids: np.ndarray) -> mx.array:
        return self.finish(tuple(ids.shape), ids)


class HyV4EngramHasher:
    """Exact Hy4-flash bigram/trigram deterministic prime-modulo hasher using checkpoint `lookup_table`.

    Contract:
    - Uses the frozen checkpoint `lookup_table` (`I32 [N_tbl]`, e.g. `120818`) directly.
    - Out-of-range padded token IDs (`tid >= len(lookup_table)`) MUST map to `lookup_table[engram_pad_id]`.
    - Multipliers use 1-based `layer_id` (`[2, 12, 22, 32]`) and `engram_seed`:
        `half_bound = (INT64_MAX // compressed_vocab_size) // 2`
        `rng = np.random.default_rng(engram_seed + 10007 * layer_id)`
        `M = rng.integers(0, half_bound, size=max_ngram_size, dtype=np.int64) * 2 + 1`
    - Rolling XOR: `z_2 = (c_0 * M[0]) ^ (c_1 * M[1])`, `z_3 = z_2 ^ (c_2 * M[2])`.
    - Non-negative Euclidean modulo `z_k % prime` + cumulative per-head table offsets.
    """

    def __init__(
        self,
        config: HyV4FlashConfig,
        lookup_table: np.ndarray,
    ) -> None:
        self.config = config
        self.max_ngram = int(config.engram_max_ngram_size)
        self.n_heads_per_ngram = int(config.engram_n_head_per_ngram)
        self.total_heads = (self.max_ngram - 1) * self.n_heads_per_ngram
        self.pad_token_id = int(config.engram_pad_id)
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)

        tbl = np.asarray(lookup_table, dtype=np.int64).reshape(-1)
        self.raw_table_len = int(tbl.shape[0])
        self.compressed_pad_id = int(tbl[self.pad_token_id])
        self.compressed_vocab_size = int(tbl.max()) + 1

        # Pad lookup_table to full vocab_size so padded IDs (120818..120831) map to compressed_pad_id!
        full_len = max(int(config.vocab_size), self.raw_table_len)
        padded = np.full(full_len, self.compressed_pad_id, dtype=np.int64)
        padded[: self.raw_table_len] = tbl
        self.lookup_table = padded

        half_bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.layer_ids_1based = [int(x) for x in config.engram_layer_ids]
        self.zero_to_one = {lid - 1: lid for lid in self.layer_ids_1based}
        self.multipliers: Dict[int, np.ndarray] = {}
        self.primes: Dict[int, np.ndarray] = {}
        self.offsets: Dict[int, np.ndarray] = {}

        for lid in self.layer_ids_1based:
            zidx = lid - 1
            rng = np.random.default_rng(int(config.engram_seed) + 10007 * lid)
            self.multipliers[zidx] = rng.integers(
                0, half_bound, size=(self.max_ngram,), dtype=np.int64
            ) * 2 + 1
            raw_primes = config.engram_layer_head_vocab_sizes.get(str(lid))
            if raw_primes is None:
                raw_primes = config.engram_layer_head_vocab_sizes.get(lid)
            if raw_primes is None:
                raise KeyError(f"Missing engram_layer_head_vocab_sizes entry for layer_id={lid}")
            p_arr = np.asarray(raw_primes, dtype=np.int64)  # [max_ngram - 1, n_heads_per_ngram]
            self.primes[zidx] = p_arr
            flat = p_arr.reshape(-1)
            offs = np.zeros(flat.shape[0], dtype=np.int64)
            offs[1:] = np.cumsum(flat[:-1])
            self.offsets[zidx] = offs

    def compress_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        """Map raw token IDs [L] -> compressed token IDs [L] (clamping out-of-range to pad)."""
        arr = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        clipped = np.where((arr >= 0) & (arr < self.lookup_table.shape[0]), arr, self.pad_token_id)
        return self.lookup_table[clipped]

    def valid_mask(self, input_ids: np.ndarray) -> np.ndarray:
        """Mask True for real text tokens, False for pad / image / video tokens."""
        arr = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        return (
            (arr != self.pad_token_id)
            & (arr != self.image_token_id)
            & (arr != self.video_token_id)
        )

    def hash_for_layers(
        self,
        input_ids: np.ndarray,
        prev_compressed: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
        """Compute hash row indices `[L, 16]` for every Engram layer + updated trailing history.

        Args:
            input_ids: 1-D raw token IDs `[L]`.
            prev_compressed: Optional last `max_ngram - 1` (2) compressed token IDs before this step.

        Returns:
            `(layer_hash_ids, new_tail_compressed, valid_mask_bool)`
        """
        comp = self.compress_tokens(input_ids)
        L = int(comp.shape[0])
        n = self.max_ngram
        pad_c = self.compressed_pad_id
        prev_len = 0 if prev_compressed is None else int(prev_compressed.shape[0])

        # Build padded window [n - 1 + L] so shift s accesses [n - 1 - s : n - 1 - s + L]
        buf = np.full(n - 1 + L, pad_c, dtype=np.int64)
        if prev_compressed is not None and prev_len > 0:
            take = min(n - 1, prev_len)
            buf[n - 1 - take : n - 1] = prev_compressed[-take:]
        buf[n - 1 :] = comp

        # Shifted token matrix [L, n]: col 0 = t, col 1 = t-1, col 2 = t-2
        toks = np.stack([buf[n - 1 - s : n - 1 - s + L] for s in range(n)], axis=-1)

        out: Dict[int, np.ndarray] = {}
        for lid in self.layer_ids_1based:
            zidx = lid - 1
            mult = self.multipliers[zidx]  # [n]
            primes = self.primes[zidx]  # [n - 1, H]
            offs = self.offsets[zidx]  # [(n - 1) * H]

            # Exact signed int64 multiplication + rolling XOR + non-negative Euclidean modulo
            with np.errstate(over="ignore"):
                prod = toks * mult[None, :]
            rolling = prod[:, 0]
            parts = []
            for k in range(1, n):
                rolling = np.bitwise_xor(rolling, prod[:, k])
                # Python/NumPy `%` with positive divisor already returns Euclidean non-negative [0, p-1]!
                parts.append(rolling[:, None] % primes[k - 1][None, :])
            out[zidx] = np.concatenate(parts, axis=-1) + offs[None, :]

        new_tail = buf[-(n - 1) :].copy() if (n - 1) > 0 else np.zeros(0, dtype=np.int64)
        return out, new_tail, self.valid_mask(input_ids)


class HyV4EngramLayer(nn.Module):
    """Hy4-flash Engram fusion module updating the 4-stream iHC state `[B, L, 4, 3072]`."""

    def __init__(self, config: HyV4FlashConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hc_mult = int(config.hc_mult)
        self.hidden_size = int(config.hidden_size)
        self.n_heads = (int(config.engram_max_ngram_size) - 1) * int(config.engram_n_head_per_ngram)
        self.head_dim = int(config.engram_n_embed_per_ngram) // int(config.engram_n_head_per_ngram)
        self.engram_dim = self.n_heads * self.head_dim  # 16 * 96 = 1536
        self.inv_sqrt_d = self.hidden_size ** -0.5

        # Per-head RMSNorms over the 16 retrieved 96-dim rows (eps = finfo(f32).eps)
        self.lookup_per_head_norms = [
            nn.RMSNorm(self.head_dim, eps=_ENGRAM_NORM_EPS) for _ in range(self.n_heads)
        ]
        # Per-iHC-stream key projections (WITH bias!) and key/query RMSNorms
        self.key_projs = [
            nn.Linear(self.engram_dim, self.hidden_size, bias=True) for _ in range(self.hc_mult)
        ]
        self.norm_key = [
            nn.RMSNorm(self.hidden_size, eps=_ENGRAM_NORM_EPS) for _ in range(self.hc_mult)
        ]
        self.norm_query = [
            nn.RMSNorm(self.hidden_size, eps=_ENGRAM_NORM_EPS) for _ in range(self.hc_mult)
        ]
        # Shared value projection (WITH bias!), pre-gamma, and per-stream bypass scales
        self.value_proj = nn.Linear(self.engram_dim, self.hidden_size, bias=True)
        self.unified_pre_gamma = mx.ones((self.hidden_size,), dtype=mx.bfloat16)
        self.unified_per_hc_scales = [
            mx.zeros((self.hidden_size,), dtype=mx.bfloat16) for _ in range(self.hc_mult)
        ]

        # Disk-backed (or in-memory test) embedding table attached by loader (NOT an nn.Module param!)
        self._table: Optional[HyV4DiskEngramTable | HyV4InMemoryEngramTable] = None
        # Stacked head norm weights [16, 96] cached on eval for single-op vectorized RMSNorm
        self._head_norm_stack: Optional[mx.array] = None

    def bind_table(self, table: HyV4DiskEngramTable | HyV4InMemoryEngramTable) -> None:
        self._table = table

    def prepare_runtime(self) -> None:
        w = mx.stack([n.weight for n in self.lookup_per_head_norms], axis=0)
        mx.eval(w)
        self._head_norm_stack = w

    def __call__(
        self,
        hidden_hc: mx.array,
        raw_embeds: mx.array,
        valid_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Apply exact Hy4-flash Engram residual fusion to 4-stream iHC tensor `[B, L, 4, D]`.

        Args:
            hidden_hc: `[B, L, 4, 3072]` BF16 4-stream iHC state.
            raw_embeds: `[B, L, 16, 96]` BF16 retrieved from disk-backed `HyV4DiskEngramTable`.
            valid_mask: Optional `[B, L]` bool (False for pad/image/video tokens).
        """
        B, L, C, D = hidden_hc.shape
        # 1. Per-head RMSNorm across all 16 heads simultaneously -> flatten to [B, L, 1536]
        if self._head_norm_stack is not None:
            normed_h = mx.fast.rms_norm(raw_embeds, None, _ENGRAM_NORM_EPS) * self._head_norm_stack
        else:
            w_stack = mx.stack([n.weight for n in self.lookup_per_head_norms], axis=0)
            normed_h = mx.fast.rms_norm(raw_embeds, None, _ENGRAM_NORM_EPS) * w_stack
        e = normed_h.reshape(B, L, self.engram_dim)

        # 2. Shared value path: main = unified_pre_gamma * unweighted_rmsnorm(value_proj(e))
        val = self.value_proj(e)
        main = mx.fast.rms_norm(val, None, _ENGRAM_NORM_EPS) * self.unified_pre_gamma

        # 3. Per-stream gated injection: gate_c = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
        mask_f = None
        if valid_mask is not None:
            mask_f = valid_mask.reshape(B, L, 1).astype(hidden_hc.dtype)

        streams = []
        for c in range(C):
            h_c = hidden_hc[:, :, c, :]
            k_c = self.norm_key[c](self.key_projs[c](e))
            q_c = self.norm_query[c](h_c)
            dot = mx.sum(k_c.astype(mx.float32) * q_c.astype(mx.float32), axis=-1, keepdims=True) * self.inv_sqrt_d
            signed_sqrt = mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
            gate_c = mx.sigmoid(signed_sqrt).astype(hidden_hc.dtype)
            delta_c = (gate_c + self.unified_per_hc_scales[c]) * main
            if mask_f is not None:
                delta_c = delta_c * mask_f
            streams.append(h_c + delta_c)
        return mx.stack(streams, axis=2)

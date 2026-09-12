# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (glm5_next) text runtime for vMLX.

Port of the parity-proven jang_tools glm5_next reference (component-level
bit-exactness vs transformers-main verified 2026-08-29: experts 1.6e-9, MLA
5.6e-5, KDA-block 3.5e-5, router identical; prefill==chunked==stepwise),
restructured onto the mlx-lm Model/ModelArgs/cache contract so the standard
loader and the vMLX serving lanes drive it.

Architecture (45 layers, hidden 4096, vocab 154880, untied lm_head):
  * 34 KDA linear-attention layers (Kimi delta rule; 64 heads x 128; one
    depthwise causal conv PER q/k/v projection with SiLU; l2-normed q/k;
    smooth sigmoid-bounded decay gate
        g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    — NOT the Ling clamped-softplus form). Fixed-size state: S [B,64,128,128]
    fp32 + 3 conv states per layer (~150 MB total) — context-independent.
  * 11 MLA layers (DeepSeek-V3 MLA, PURE NoPE — qk_rope_head_dim=0, no rotary
    anywhere; q_lora 1536 / kv_lora 512; 64 heads x 256qk/256v) at every 4th
    position. The DSA indexer selects compressed key pools beyond index_topk;
    below that boundary dense causal attention is bit-exact.
  * mHC hyper-connections with 20-iteration Sinkhorn per sublayer (4 streams,
    unweighted-mean final collapse). ALL mHC math in fp32.
  * MoE layers 3-44: 288 routed experts top-8 (sigmoid scores +
    e_score_correction_bias, n_group=1), norm_topk_prob, scaling 2.5, shared
    expert; clamped SwiGLU (±10) EVERYWHERE incl. dense layers 0-2.
  * The optional layer-45 MTP block is a non-mHC MLA+MoE decoder fed by
    eh_proj(concat(enorm(next-token embedding), hnorm(previous hidden))). Its
    private shared-head norm precedes the model's shared LM head.
  * fp32 storage keeps: A_log, dt_bias, e_score_correction_bias, hc_base,
    hc_scale. fp32 COMPUTE regardless of storage: router logits+topk, mHC,
    KDA gate/beta/recurrence, l2norm, o_norm, SDPA (the L==1 MLA-absorb
    trap), final rms norms.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-29.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.switch_layers import SwitchGLU

from vmlx_engine.metal.affine_moe_pair_decode import (
    affine_moe_routed_output,
    install_affine_moe_pair_decode,
)

from vmlx_engine.metal.gated_rmsnorm_decode import (
    fused_gated_rmsnorm_requested,
    sigmoid_gated_rmsnorm_small_rows,
)
from vmlx_engine.metal.glm5_mhc_prefill import (
    fused_glm5_mhc_prefill_requested,
    glm5_mhc_prefill_sinkhorn,
)
from vmlx_engine.metal.glm5_mhc_decode import (
    fused_glm5_mhc_requested,
    glm5_mhc_decode,
)
from vmlx_engine.metal.glm5_hc_place_decode import (
    fused_glm5_hc_place_requested,
    glm5_hc_place_decode,
)
from vmlx_engine.metal.kda_conv_decode import (
    fused_kda_conv_requested,
    glm5_kda_conv_decode,
)
from vmlx_engine.metal.glm5_kda_conv_prefill import kda_conv_prefill
from vmlx_engine.metal.glm5_kda_qkv_prefill import (
    kda_qkv_prefill,
    qkv_prefill_requested,
)
from vmlx_engine.metal.kda_step_decode import (
    fused_kda_step_requested,
    glm5_kda_step_decode,
)

from vmlx_engine.metal.quantized_projection_group import (
    QuantizedProjectionGroup,
    quantized_projection_group_reason,
)
from vmlx_engine.metal.sparse_index_score_decode import (
    fused_sparse_index_score_requested,
    sparse_index_scores_decode,
)

try:  # package import (registered under mlx_lm.models.glm5_next)
    from vmlx_engine.models.glm5_next.kda import (
        kda_chunked,
        kda_recurrent,
        kda_recurrent_with_states,
        kda_step,
        l2norm,
        short_conv,
        short_conv_with_states,
    )
except ImportError:  # direct file execution fallback
    from kda import (  # type: ignore
        kda_chunked,
        kda_recurrent,
        kda_recurrent_with_states,
        kda_step,
        l2norm,
        short_conv,
        short_conv_with_states,
    )

# ArraysCache slot layout for KDA layers.
KDA_CONV_Q = 0
KDA_CONV_K = 1
KDA_CONV_V = 2
KDA_STATE = 3

# Glm5MLACache slot layout for MLA layers.
MLA_KEYS = 0
MLA_VALUES = 1
MLA_PACKED = 2  # indexer packed history: [B, T, head_dim(k) + head_dim(gate)]
MLA_POOL_KEYS = 3  # completed DSA k-pools: [B, floor(T / kpool), head_dim]

# Absorbed Glm5MLACache slot layout.  Keeping the same concrete cache class
# lets the scheduler's typed prompt-cache ownership remain unchanged while the
# schema tag below prevents an expanded-v1 SSD record from being mistaken for
# an absorbed-v2 record.
MLA_LATENT = 0  # shared normalized kv_a latent: [B, 1, T, kv_lora_rank]
MLA_LATENT_PACKED = 1  # DSA key + pool-gate history: [B, T, 2 * index_dim]
MLA_LATENT_POOL_KEYS = 2  # completed pool-4 DSA keys
MLA_LATENT_RESERVED = 3  # safetensors-stable reserved zero-length tensor

# Compact MLA state grows in bounded blocks instead of rebuilding the complete
# history for every decoded token.  2,048 is the architecture's dense/sparse
# transition and the normal prefill quantum; the pool-key block is derived from
# the bundle's k-pool ratio.  Unlike geometric doubling, block growth keeps the
# spare allocation bounded beside a roughly 95GB model.
_GLM5_MLA_CAPACITY_TOKENS = 2_048

_LOG = logging.getLogger("vmlx_engine")
_GLM5_MLA_PATHS_LOGGED: set[str] = set()


def glm5_mla_absorb_enabled() -> bool:
    """Return whether the compact MLA cache is enabled.

    Compact v2 is the GLM-5.3 default after an exact 5,268-token live A/B:
    expanded v1 retained 3.826 GiB per prompt and approached the wired limit,
    while v2 retained 261 MiB and preserved correct tool/cache/refault
    behaviour.  Keep an explicit environment opt-out for diagnostics and
    numerical comparisons with the legacy expanded representation.
    """

    return os.environ.get(
        "VMLINUX_GLM5_MLA_ABSORB",
        os.environ.get("VMLX_GLM5_MLA_ABSORB", "1"),
    ).strip().lower() in {"1", "true", "yes", "on"}


class Glm5KDACache(ArraysCache):
    """KDA recurrent state with exact partial speculative rollback.

    A multi-draft verify forward commits one target token and speculatively
    advances over N draft tokens.  Unlike a KV cache, KDA cannot be rewound by
    trimming a token offset: every token mutates the recurrent matrix and the
    three causal-convolution tails.  The GLM attention layer therefore records
    the state after each verify position.  A partial rejection can restore the
    state after the confirmed token plus the accepted draft prefix.
    """

    supports_partial_rollback = True

    def __init__(self):
        super().__init__(4)
        self._speculative_states: list[tuple[mx.array, ...]] | None = None

    def set_speculative_states(self, states: list[tuple[mx.array, ...]]) -> None:
        self._speculative_states = states

    def rollback_speculative(self, rejected: int) -> bool:
        states = self._speculative_states
        if not states or rejected < 0 or rejected >= len(states):
            return False
        accepted_drafts = len(states) - 1 - rejected
        self.cache = list(states[accepted_drafts])
        self._speculative_states = None
        return True

    def commit_speculative(self) -> None:
        self._speculative_states = None

    def extract(self, idx: int) -> "Glm5KDACache":
        """Extract one batch row without indexing valid empty state slots.

        ``mlx_lm.ArraysCache.extract`` assumes every slot is an array and
        returns a plain ``ArraysCache``.  GLM KDA slots are intentionally
        populated lazily, so a terminal BatchGenerator response can still
        contain ``None`` here.  Preserve both that state and the concrete
        rollback-aware cache type when handing the finished request back to
        the scheduler.
        """

        extracted = Glm5KDACache()
        extracted.cache = [
            value[idx : idx + 1] if value is not None else None
            for value in self.cache
        ]
        return extracted

    @classmethod
    def merge(cls, caches) -> "Glm5KDACache":
        """Batch KDA rows without degrading to a generic ``ArraysCache``."""

        return _merge_glm5_arrays_caches(caches, cls())

    @property
    def meta_state(self):
        return ("glm5_next_native_v1", "kda")

    @meta_state.setter
    def meta_state(self, value):
        meta = tuple(value or ())
        if meta != ("glm5_next_native_v1", "kda"):
            raise ValueError("GLM KDA cache metadata is missing its typed schema")

    @classmethod
    def from_state(cls, state, meta_state) -> "Glm5KDACache":
        meta = tuple(meta_state or ())
        if meta != ("glm5_next_native_v1", "kda"):
            raise ValueError("GLM KDA cache metadata is missing its typed schema")
        values = list(state or ())
        if len(values) != 4:
            raise ValueError("GLM KDA cache must contain exactly four state arrays")
        populated = [value is not None for value in values]
        if any(populated) and not all(populated):
            raise ValueError("GLM KDA cache has a partial recurrent-state record")
        if all(populated):
            if any(getattr(value, "ndim", -1) != 3 for value in values[:3]):
                raise ValueError("GLM KDA convolution states must be rank 3")
            if getattr(values[KDA_STATE], "ndim", -1) != 4:
                raise ValueError("GLM KDA recurrent state must be rank 4")
            batch = int(values[0].shape[0])
            if any(int(value.shape[0]) != batch for value in values[1:]):
                raise ValueError("GLM KDA state arrays disagree on batch size")
        rebuilt = cls()
        rebuilt.cache = values
        rebuilt._speculative_states = None
        return rebuilt


class Glm5MLACache(ArraysCache):
    """Typed MLA/DSA cache with materialized-v1 and absorbed-v2 schemas.

    The legacy representation stores expanded per-head K/V.  The experimental
    absorbed representation stores the shared normalized ``kv_a`` latent plus
    the DSA indexer's packed history and completed pool keys.  Its physical
    buffers grow in bounded blocks while ``cache`` and ``state`` expose only
    the logical prefix.  Prompt-boundary wrappers use per-slot copy-on-write,
    preserving exact N-1 snapshots even when decode or rollback reuses spare
    capacity.
    """

    def __init__(self, kpool: int = 4, *, absorbed: bool | None = None):
        super().__init__(4)
        if int(kpool) <= 0:
            raise ValueError("GLM DSA kpool must be positive")
        self.kpool = int(kpool)
        self.absorbed = (
            glm5_mla_absorb_enabled() if absorbed is None else bool(absorbed)
        )
        self._capacity_buffers: list[mx.array | None] = [None] * 4
        self._capacity_lengths = [0] * 4
        self._capacity_shared = [False] * 4

    @staticmethod
    def _capacity_axis(slot: int) -> int:
        if slot == MLA_LATENT:
            return 2
        if slot in {MLA_LATENT_PACKED, MLA_LATENT_POOL_KEYS}:
            return 1
        raise ValueError(f"GLM absorbed MLA slot {slot} has no sequence axis")

    def _capacity_block(self, slot: int) -> int:
        if slot == MLA_LATENT_POOL_KEYS:
            return max(1, _GLM5_MLA_CAPACITY_TOKENS // int(self.kpool))
        return _GLM5_MLA_CAPACITY_TOKENS

    @staticmethod
    def _logical_slice(value: mx.array, axis: int, length: int) -> mx.array:
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(0, int(length))
        return value[tuple(slices)]

    def _reset_absorbed_capacity(self) -> None:
        """Adopt current logical arrays as exact-fit physical buffers."""

        self._capacity_buffers = [None] * 4
        self._capacity_lengths = [0] * 4
        self._capacity_shared = [False] * 4
        if not self.absorbed:
            return
        for slot in (MLA_LATENT, MLA_LATENT_PACKED, MLA_LATENT_POOL_KEYS):
            value = self.cache[slot]
            if value is None or not getattr(value, "size", 0):
                continue
            axis = self._capacity_axis(slot)
            self._capacity_buffers[slot] = value
            self._capacity_lengths[slot] = int(value.shape[axis])

    def _mark_absorbed_capacity_shared(self) -> None:
        if not self.absorbed:
            return
        for slot in (MLA_LATENT, MLA_LATENT_PACKED, MLA_LATENT_POOL_KEYS):
            if self._capacity_buffers[slot] is not None:
                self._capacity_shared[slot] = True

    def _append_absorbed_capacity(
        self,
        slot: int,
        value: mx.array,
    ) -> mx.array:
        """Append to one logical state lane without per-token concatenation."""

        axis = self._capacity_axis(slot)
        count = int(value.shape[axis])
        buffer = self._capacity_buffers[slot]
        logical = int(self._capacity_lengths[slot])

        if buffer is None:
            self._capacity_buffers[slot] = value
            self._capacity_lengths[slot] = count
            self._capacity_shared[slot] = False
            self.cache[slot] = value
            return value
        if count == 0:
            return self.cache[slot]
        if value.ndim != buffer.ndim:
            raise ValueError("GLM absorbed MLA append rank differs from cache")
        for dim, (current, incoming) in enumerate(zip(buffer.shape, value.shape)):
            if dim != axis and int(current) != int(incoming):
                raise ValueError("GLM absorbed MLA append shape differs from cache")

        # An identity clone or extracted batch row can share the physical
        # allocation with a retained prompt boundary.  Detach only the lane
        # being written; mx.array(array) is a real copy in MLX 0.32.x.
        if self._capacity_shared[slot]:
            buffer = mx.array(self._logical_slice(buffer, axis, logical))
            self._capacity_buffers[slot] = buffer
            self._capacity_shared[slot] = False

        needed = logical + count
        capacity = int(buffer.shape[axis])
        if needed > capacity:
            block = self._capacity_block(slot)
            capacity = ((needed + block - 1) // block) * block
            shape = list(buffer.shape)
            shape[axis] = capacity
            grown = mx.zeros(shape, dtype=buffer.dtype)
            if logical:
                destination = [slice(None)] * grown.ndim
                destination[axis] = slice(0, logical)
                grown[tuple(destination)] = self._logical_slice(
                    buffer,
                    axis,
                    logical,
                )
            buffer = grown
            self._capacity_buffers[slot] = buffer

        destination = [slice(None)] * buffer.ndim
        destination[axis] = slice(logical, needed)
        buffer[tuple(destination)] = value
        self._capacity_lengths[slot] = needed
        logical_view = self._logical_slice(buffer, axis, needed)
        self.cache[slot] = logical_view
        return logical_view

    def extract(self, idx: int) -> "Glm5MLACache":
        """Extract one batch row while retaining MLA/DSA cache semantics."""

        extracted = Glm5MLACache(self.kpool, absorbed=self.absorbed)
        extracted.cache = [
            value[idx : idx + 1] if value is not None else None
            for value in self.cache
        ]
        if self.absorbed:
            extracted._reset_absorbed_capacity()
            self._mark_absorbed_capacity_shared()
            extracted._mark_absorbed_capacity_shared()
        return extracted

    @classmethod
    def merge(cls, caches) -> "Glm5MLACache":
        """Batch MLA/DSA rows while preserving the shared k-pool contract."""

        kpools = {int(cache.kpool) for cache in caches}
        if len(kpools) != 1:
            raise ValueError("GLM MLA caches disagree on DSA kpool")
        modes = {bool(cache.absorbed) for cache in caches}
        if len(modes) != 1:
            raise ValueError("GLM MLA caches disagree on absorbed representation")
        merged = _merge_glm5_arrays_caches(
            caches,
            cls(kpools.pop(), absorbed=modes.pop()),
        )
        if merged.absorbed:
            merged._reset_absorbed_capacity()
            if len(caches) == 1:
                caches[0]._mark_absorbed_capacity_shared()
                merged._mark_absorbed_capacity_shared()
        return merged

    @property
    def state(self):
        """Return a safetensors-safe representation of the typed MLA state.

        Dense-only MLA prompts legitimately have no materialized DSA pool even
        after KV/index history exists.  ``mlx.utils.tree_flatten`` retains that
        ``None`` leaf, but safetensors accepts tensors only.  Encode the absent
        pool as a typed zero-length tensor; ``from_state`` accepts this as the
        exact not-yet-materialized state.
        """

        values = list(self.cache)
        if self.absorbed:
            latent = values[MLA_LATENT]
            if latent is None:
                empty = mx.zeros((0,), dtype=mx.bfloat16)
                return [empty, empty, empty, empty]
            packed = values[MLA_LATENT_PACKED]
            if packed is None:
                raise ValueError("GLM absorbed MLA cache is missing DSA history")
            pool = values[MLA_LATENT_POOL_KEYS]
            if pool is None:
                pool = mx.zeros(
                    (int(latent.shape[0]), 0, int(packed.shape[-1]) // 2),
                    dtype=packed.dtype,
                )
            reserved = mx.zeros((0,), dtype=latent.dtype)
            return [latent, packed, pool, reserved]
        packed = values[MLA_PACKED]
        if packed is not None and values[MLA_POOL_KEYS] is None:
            packed_width = int(packed.shape[-1])
            if packed_width <= 0 or packed_width % 2:
                raise ValueError(
                    "GLM MLA packed index width must be positive and even"
                )
            values[MLA_POOL_KEYS] = mx.zeros(
                (int(packed.shape[0]), 0, packed_width // 2),
                dtype=packed.dtype,
            )
        return values

    @state.setter
    def state(self, value):
        self.cache = list(value)
        self._reset_absorbed_capacity()

    def filter(self, batch_indices):
        super().filter(batch_indices)
        self._reset_absorbed_capacity()

    def extend(self, other):
        super().extend(other)
        self._reset_absorbed_capacity()

    @property
    def nbytes(self):
        if not self.absorbed:
            return super().nbytes
        return sum(
            value.nbytes
            for value in self._capacity_buffers
            if value is not None
        )

    @property
    def offset(self) -> int:
        first = self.cache[MLA_LATENT if self.absorbed else MLA_KEYS]
        if first is None or getattr(first, "size", 0) == 0:
            return 0
        return int(first.shape[2])

    @property
    def pool_keys(self) -> mx.array | None:
        slot = MLA_LATENT_POOL_KEYS if self.absorbed else MLA_POOL_KEYS
        value = self.cache[slot]
        return value if value is not None and getattr(value, "size", 0) else None

    def update_latent(self, latent: mx.array) -> mx.array:
        if not self.absorbed:
            raise ValueError("latent update requires absorbed GLM MLA cache")
        return self._append_absorbed_capacity(MLA_LATENT, latent)

    def update_kv(self, k: mx.array, v: mx.array):
        if self.absorbed:
            raise ValueError("expanded KV update requires materialized GLM MLA cache")
        if self.cache[MLA_KEYS] is not None:
            k = mx.concatenate([self.cache[MLA_KEYS], k], axis=2)
            v = mx.concatenate([self.cache[MLA_VALUES], v], axis=2)
        self.cache[MLA_KEYS] = k
        self.cache[MLA_VALUES] = v
        return k, v

    def update_packed(self, packed: mx.array) -> mx.array:
        slot = MLA_LATENT_PACKED if self.absorbed else MLA_PACKED
        if self.absorbed:
            return self._append_absorbed_capacity(slot, packed)
        if self.cache[slot] is not None:
            packed = mx.concatenate([self.cache[slot], packed], axis=1)
        self.cache[slot] = packed
        return packed

    def update_pool_keys(self, pool_keys: mx.array) -> mx.array:
        slot = MLA_LATENT_POOL_KEYS if self.absorbed else MLA_POOL_KEYS
        if self.absorbed:
            return self._append_absorbed_capacity(slot, pool_keys)
        existing = self.cache[slot]
        if existing is not None:
            pool_keys = mx.concatenate([existing, pool_keys], axis=1)
        self.cache[slot] = pool_keys
        return pool_keys

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = max(0, min(int(n), self.offset))
        if n == 0:
            return 0
        keep = self.offset - n
        if self.absorbed:
            for slot, length in (
                (MLA_LATENT, keep),
                (MLA_LATENT_PACKED, keep),
                (MLA_LATENT_POOL_KEYS, keep // int(self.kpool)),
            ):
                buffer = self._capacity_buffers[slot]
                if buffer is None:
                    continue
                axis = self._capacity_axis(slot)
                self._capacity_lengths[slot] = int(length)
                self.cache[slot] = self._logical_slice(buffer, axis, length)
            return n
        if self.cache[MLA_KEYS] is not None:
            self.cache[MLA_KEYS] = self.cache[MLA_KEYS][..., :keep, :]
            self.cache[MLA_VALUES] = self.cache[MLA_VALUES][..., :keep, :]
            self.cache[MLA_PACKED] = self.cache[MLA_PACKED][:, :keep, :]
            if self.cache[MLA_POOL_KEYS] is not None:
                # A compressed pool is valid only when all of its raw tokens
                # remain.  A future append recomputes the newly completed pool
                # from MLA_PACKED after speculative rollback.
                keep_pools = keep // int(self.kpool)
                self.cache[MLA_POOL_KEYS] = self.cache[MLA_POOL_KEYS][
                    :, :keep_pools, :
                ]
        return n

    @property
    def meta_state(self):
        if self.absorbed:
            return (
                "glm5_next_native_v2",
                "mla_absorbed",
                str(self.kpool),
                str(self.offset),
            )
        return (
            "glm5_next_native_v1",
            "mla",
            str(self.kpool),
            str(self.offset),
        )

    @meta_state.setter
    def meta_state(self, value):
        # Validation and constructor invariants are owned by from_state().
        meta = tuple(value or ())
        if len(meta) != 4 or meta[:2] not in {
            ("glm5_next_native_v1", "mla"),
            ("glm5_next_native_v2", "mla_absorbed"),
        }:
            raise ValueError("GLM MLA cache metadata is missing its typed schema")
        self.absorbed = meta[:2] == ("glm5_next_native_v2", "mla_absorbed")

    @classmethod
    def from_state(cls, state, meta_state) -> "Glm5MLACache":
        meta = tuple(meta_state or ())
        valid_prefixes = {
            ("glm5_next_native_v1", "mla"),
            ("glm5_next_native_v2", "mla_absorbed"),
        }
        if len(meta) != 4 or meta[:2] not in valid_prefixes:
            raise ValueError("GLM MLA cache metadata is missing its typed schema")
        absorbed = meta[:2] == ("glm5_next_native_v2", "mla_absorbed")
        try:
            kpool = int(meta[2])
            expected_offset = int(meta[3])
        except (TypeError, ValueError) as exc:
            raise ValueError("GLM MLA cache metadata is not integral") from exc
        if kpool <= 0 or expected_offset < 0:
            raise ValueError("GLM MLA cache metadata is outside its valid range")
        values = list(state or ())
        if len(values) != 4:
            raise ValueError("GLM MLA cache must contain exactly four state arrays")
        if absorbed:
            latent, packed, pool, reserved = values
            if expected_offset == 0:
                if any(getattr(value, "size", 0) for value in values):
                    raise ValueError(
                        "GLM absorbed MLA empty cache carries populated state"
                    )
                return cls(kpool, absorbed=True)
            if getattr(latent, "ndim", -1) != 4 or int(latent.shape[1]) != 1:
                raise ValueError("GLM absorbed MLA latent must be [B,1,T,R]")
            if getattr(packed, "ndim", -1) != 3:
                raise ValueError("GLM absorbed MLA DSA history must be rank 3")
            if int(packed.shape[-1]) <= 0 or int(packed.shape[-1]) % 2:
                raise ValueError("GLM absorbed MLA DSA width must be positive and even")
            lengths = (int(latent.shape[2]), int(packed.shape[1]))
            if lengths != (expected_offset, expected_offset):
                raise ValueError(
                    "GLM absorbed MLA latent/index lengths do not match typed offset"
                )
            if int(latent.shape[0]) != int(packed.shape[0]):
                raise ValueError("GLM absorbed MLA state arrays disagree on batch size")
            if getattr(pool, "ndim", -1) != 3:
                raise ValueError("GLM absorbed MLA pool keys must be rank 3")
            expected_pool_length = expected_offset // kpool
            if int(pool.shape[1]) not in (0, expected_pool_length):
                raise ValueError(
                    "GLM absorbed MLA pool length is neither absent nor offset/kpool"
                )
            if int(pool.shape[0]) != int(latent.shape[0]):
                raise ValueError("GLM absorbed MLA pool keys disagree on batch size")
            if getattr(reserved, "size", -1) != 0:
                raise ValueError("GLM absorbed MLA reserved state must be empty")
            rebuilt = cls(kpool, absorbed=True)
            rebuilt.state = [
                latent,
                packed,
                pool if int(pool.shape[1]) else None,
                None,
            ]
            return rebuilt
        populated = [value is not None for value in values]
        if any(populated) and not all(populated[:3]):
            raise ValueError("GLM MLA cache is missing KV or packed index state")
        if all(populated[:3]):
            keys, vals, packed = values[:3]
            if getattr(keys, "ndim", -1) != 4 or getattr(vals, "ndim", -1) != 4:
                raise ValueError("GLM MLA keys and values must be rank 4")
            if getattr(packed, "ndim", -1) != 3:
                raise ValueError("GLM MLA packed index history must be rank 3")
            lengths = (int(keys.shape[2]), int(vals.shape[2]), int(packed.shape[1]))
            if lengths != (expected_offset, expected_offset, expected_offset):
                raise ValueError(
                    "GLM MLA KV/index lengths do not match the typed offset"
                )
            pool = values[MLA_POOL_KEYS]
            if pool is not None:
                if getattr(pool, "ndim", -1) != 3:
                    raise ValueError("GLM MLA pool keys must be rank 3")
                if int(pool.shape[0]) != int(keys.shape[0]):
                    raise ValueError("GLM MLA pool keys disagree on batch size")
                expected_pool_length = expected_offset // kpool
                if int(pool.shape[1]) not in (0, expected_pool_length):
                    raise ValueError(
                        "GLM MLA pool length is neither absent nor offset/kpool"
                    )
        elif expected_offset != 0:
            raise ValueError("GLM MLA empty cache cannot carry a non-zero offset")
        rebuilt = cls(kpool, absorbed=False)
        rebuilt.cache = values
        return rebuilt


def _merge_glm5_arrays_caches(caches, merged):
    """Typed equivalent of mlx-lm ``ArraysCache.merge``.

    Upstream constructs the result as ``cls(n_state)``, which is invalid for
    both GLM cache constructors and would also discard MLA's k-pool metadata.
    """

    if not caches:
        return merged
    batch_size = len(caches)
    if batch_size == 1:
        # GLM is deliberately scheduled one request at a time. Its cache
        # arrays are functionally replaced on update, so a new typed wrapper
        # can take ownership of the existing immutable boundary without a
        # model-sized zeros+assignment copy before the final prompt token.
        # ``state`` is the safetensors serialization view.  An empty absorbed
        # cache encodes its absent slots as rank-1 zero-length tensors there;
        # feeding those sentinels back into the live cache makes the first
        # latent update try to concatenate rank 1 with rank 4.  Runtime merge
        # owns the raw cache slots, never the persistence representation.
        merged.cache = list(caches[0].cache)
        merged.left_padding = getattr(caches[0], "left_padding", None)
        merged.lengths = getattr(caches[0], "lengths", None)
        return merged
    if all(cache.empty() for cache in caches):
        merged.left_padding = mx.array([0] * batch_size)
        return merged

    for state_index in range(len(merged.cache)):
        populated = [
            cache.cache[state_index]
            for cache in caches
            if cache.cache[state_index] is not None
            and getattr(cache.cache[state_index], "size", 0)
        ]
        if not populated:
            continue
        exemplar = populated[0]
        shape = list(exemplar.shape)
        shape[0] = batch_size
        value = mx.zeros(shape, exemplar.dtype)
        for batch_index, cache in enumerate(caches):
            source = cache.cache[state_index]
            if source is not None and getattr(source, "size", 0):
                value[batch_index : batch_index + 1] = source
        merged.cache[state_index] = value
    return merged


def clone_glm5_next_layer_cache(
    source: Glm5KDACache | Glm5MLACache,
    *,
    copy_fn: Callable[[mx.array], mx.array],
) -> Glm5KDACache | Glm5MLACache:
    """Return an isolated copy of one exact GLM native-state boundary."""

    if type(source).__name__ not in {"Glm5KDACache", "Glm5MLACache"}:
        raise TypeError(f"unexpected GLM cache class: {type(source).__name__}")
    source_state = list(source.state)
    copied_state = [
        None if value is None else copy_fn(value) for value in source_state
    ]
    cloned = type(source).from_state(copied_state, source.meta_state)
    if isinstance(source, Glm5MLACache) and source.absorbed:
        shares_storage = any(
            original is copied
            for original, copied in zip(source_state, copied_state)
            if original is not None
        )
        if shares_storage:
            source._mark_absorbed_capacity_shared()
            cloned._mark_absorbed_capacity_shared()
    return cloned


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm5_next"
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    rms_norm_eps: float = 1e-5
    vocab_size: int = 154880
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # KDA
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel: int = 4
    linear_lower_bound: float = -5.0
    # MLA
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    index_topk: int = 2048
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    max_position_embeddings: int = 1_048_576
    # MoE
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    intermediate_size: int = 12288
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    first_k_dense_replace: int = 3
    layer_types: List[str] = field(default_factory=list)
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        t = dict(params.get("text_config") or params)
        la = t.get("linear_attn_config") or {}
        if t.get("n_group", 1) != 1 or t.get("topk_group", 1) != 1:
            raise ValueError("glm5_next runtime implements n_group=1 routing only")
        if (t.get("qk_rope_head_dim") or 0) != 0:
            raise ValueError("glm5_next runtime expects pure-NoPE MLA (qk_rope_head_dim=0)")
        merged = {
            "model_type": params.get("model_type", "glm5_next"),
            "hidden_size": t["hidden_size"],
            "num_hidden_layers": t["num_hidden_layers"],
            "rms_norm_eps": t.get("rms_norm_eps", 1e-5),
            "vocab_size": t["vocab_size"],
            "hc_mult": t.get("hc_mult", 4),
            "hc_sinkhorn_iters": t.get("hc_sinkhorn_iters", 20),
            "hc_eps": t.get("hc_eps", 1e-6),
            "linear_num_heads": la.get("num_heads", 64),
            "linear_head_dim": la.get("head_dim", 128),
            "linear_conv_kernel": la.get("short_conv_kernel_size", 4),
            "linear_lower_bound": la.get("gate_lower_bound", -5.0),
            "num_attention_heads": t["num_attention_heads"],
            "q_lora_rank": t["q_lora_rank"],
            "kv_lora_rank": t["kv_lora_rank"],
            "qk_nope_head_dim": t["qk_nope_head_dim"],
            "v_head_dim": t["v_head_dim"],
            "index_topk": t.get("index_topk", 2048),
            "index_n_heads": t.get("index_n_heads", 32),
            "index_head_dim": t.get("index_head_dim", 128),
            "index_kpool": t.get("index_kpool", 4),
            "index_kpool_always_select_tail": t.get(
                "index_kpool_always_select_tail", True
            ),
            "max_position_embeddings": t.get("max_position_embeddings", 1_048_576),
            "n_routed_experts": t["n_routed_experts"],
            "num_experts_per_tok": t["num_experts_per_tok"],
            "moe_intermediate_size": t["moe_intermediate_size"],
            "n_shared_experts": t.get("n_shared_experts", 1),
            "intermediate_size": t["intermediate_size"],
            "routed_scaling_factor": t.get("routed_scaling_factor", 2.5),
            "norm_topk_prob": t.get("norm_topk_prob", True),
            "swiglu_limit": t.get("swiglu_limit", 10.0),
            "first_k_dense_replace": t.get("first_k_dense_replace", 3),
            "layer_types": list(t["layer_types"]),
            "num_nextn_predict_layers": int(
                t.get("num_nextn_predict_layers", 0) or 0
            ),
            "index_share_for_mtp_iteration": bool(
                t.get("index_share_for_mtp_iteration", False)
            ),
        }
        return cls(**merged)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------- mHC ------
class HyperConnection(nn.Module):
    """fn/base/scale -> (post, comb, collapsed). All mix math in fp32."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        h, d = args.hc_mult, args.hidden_size
        self.hc_mult = h
        self.iters = args.hc_sinkhorn_iters
        self.eps = args.hc_eps
        self.rms_eps = args.rms_norm_eps
        self.hc_fn = mx.zeros(((2 + h) * h, h * d))
        self.hc_base = mx.zeros(((2 + h) * h,))
        self.hc_scale = mx.ones((3,))
        self._fused_decode = fused_glm5_mhc_requested()
        self._fused_prefill = fused_glm5_mhc_prefill_requested()

    def __call__(self, streams: mx.array):
        # streams: [B, S, H, D]
        h = self.hc_mult
        fused = glm5_mhc_decode(
            streams,
            self.hc_fn,
            self.hc_base,
            self.hc_scale,
            rms_eps=self.rms_eps,
            sink_eps=self.eps,
            iterations=self.iters,
            enabled=self._fused_decode,
        )
        if fused is not None:
            return fused
        flat = streams.reshape(*streams.shape[:2], -1).astype(mx.float32)
        flat = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + self.rms_eps)
        mix = flat @ self.hc_fn.astype(mx.float32).T
        pre_w, post_w, comb_w = mx.split(mix, [h, 2 * h], axis=-1)
        base = self.hc_base.astype(mx.float32)
        s0, s1, s2 = self.hc_scale.astype(mx.float32)

        pre = mx.sigmoid(pre_w * s0 + base[:h]) + self.eps
        post = 2.0 * mx.sigmoid(post_w * s1 + base[h:2 * h])
        comb = mx.softmax(comb_w.reshape(*comb_w.shape[:-1], h, h) * s2
                          + base[2 * h:].reshape(h, h), axis=-1) + self.eps
        fused_comb = glm5_mhc_prefill_sinkhorn(
            comb, sink_eps=self.eps, iterations=self.iters, enabled=self._fused_prefill
        )
        if fused_comb is not None:
            comb = fused_comb
        else:
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.eps)
            for _ in range(self.iters - 1):
                comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + self.eps)
                comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.eps)

        collapsed = mx.sum(pre[..., None] * streams.astype(mx.float32), axis=2)
        return post, comb, collapsed.astype(streams.dtype)


def hc_place(
    post: mx.array,
    comb: mx.array,
    out: mx.array,
    residual: mx.array,
    *,
    fused_decode: bool = False,
) -> mx.array:
    """streams' = post⊗out + combᵀ @ residual   (all [B,S,·] shapes)."""
    fused = glm5_hc_place_decode(
        post, comb, out, residual, enabled=fused_decode
    )
    if fused is not None:
        return fused
    dt = residual.dtype
    return (post.astype(dt)[..., None] * out[..., None, :]
            + mx.matmul(comb.astype(dt).transpose(0, 1, 3, 2), residual))


# ---------------------------------------------------------------- KDA ------
class KDAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        d, H, K = args.hidden_size, args.linear_num_heads, args.linear_head_dim
        qkv = H * K
        self.H, self.K = H, K
        self.lower_bound = args.linear_lower_bound
        self.q_proj = nn.Linear(d, qkv, bias=False)
        self.k_proj = nn.Linear(d, qkv, bias=False)
        self.v_proj = nn.Linear(d, qkv, bias=False)
        self.q_conv1d = mx.zeros((qkv, args.linear_conv_kernel))
        self.k_conv1d = mx.zeros((qkv, args.linear_conv_kernel))
        self.v_conv1d = mx.zeros((qkv, args.linear_conv_kernel))
        self.b_proj = nn.Linear(d, H, bias=False)
        self.f_a_proj = nn.Linear(d, K, bias=False)
        self.f_b_proj = nn.Linear(K, qkv, bias=False)
        self.g_a_proj = nn.Linear(d, K, bias=False)
        self.g_b_proj = nn.Linear(K, qkv, bias=False)
        self.A_log = mx.zeros((H,))
        self.dt_bias = mx.zeros((qkv,))
        self.o_norm = mx.ones((K,))
        self.o_proj = nn.Linear(qkv, d, bias=False)
        self.rms_eps = args.rms_norm_eps
        self.qkv_group = None
        self._fused_gated_norm = fused_gated_rmsnorm_requested()
        self._fused_kda_conv = fused_kda_conv_requested()
        self._fused_kda_prefill = os.environ.get("VMLX_GLM5_KDA_CONV_PREFILL", "0") == "1"
        self._fused_kda_prefill_observed = False
        self._fused_kda_qkv_prefill = qkv_prefill_requested()
        self._fused_kda_step = fused_kda_step_requested()
        self._vectorized_speculative_verify = os.environ.get(
            "VMLINUX_GLM5_VECTOR_KDA_VERIFY",
            os.environ.get("VMLX_GLM5_VECTOR_KDA_VERIFY", "1"),
        ).strip().lower() in {"1", "true", "yes", "on"}

    def prepare_runtime(self) -> bool:
        """Group q/k/v packed rows once and release superseded references."""

        linears = (self.q_proj, self.k_proj, self.v_proj)
        if quantized_projection_group_reason(linears) is not None:
            return False
        group = QuantizedProjectionGroup(linears)
        mx.eval(group.weight, group.scales, group.biases)
        self.qkv_group = group
        self.q_proj = self.k_proj = self.v_proj = None
        return True

    def _project_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if self.qkv_group is not None:
            q, k, v = self.qkv_group(x)
            return q, k, v
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _gate(self, x: mx.array) -> mx.array:
        # lower_bound * sigmoid(exp(A_log) * (f + dt_bias)) — smooth (-5, 0).
        B, T, _ = x.shape
        f = self.f_b_proj(self.f_a_proj(x)).astype(mx.float32) + self.dt_bias.astype(mx.float32)
        f = f.reshape(B, T, self.H, self.K)
        rate = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.H, 1)
        return self.lower_bound * mx.sigmoid(rate * f)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[ArraysCache] = None,
        n_confirmed: int = 0,
    ):
        B, T, _ = x.shape
        H, K = self.H, self.K
        conv_q = conv_k = conv_v = state = None
        if cache is not None and cache.cache[KDA_STATE] is not None:
            conv_q = cache.cache[KDA_CONV_Q]
            conv_k = cache.cache[KDA_CONV_K]
            conv_v = cache.cache[KDA_CONV_V]
            state = cache.cache[KDA_STATE]

        def run_segment(seg, cq0, ck0, cv0, s0):
            seg_t = seg.shape[1]
            q, k, v = self._project_qkv(seg)
            fused_conv = glm5_kda_conv_decode(
                q,
                k,
                v,
                cq0,
                ck0,
                cv0,
                self.q_conv1d,
                self.k_conv1d,
                self.v_conv1d,
                enabled=self._fused_kda_conv,
            )
            if fused_conv is not None:
                q, k, v, cq1, ck1, cv1 = fused_conv
            else:
                prepared = None
                preparation_mode = "separate_sum"
                if self._fused_kda_qkv_prefill and seg_t > 1:
                    prepared = kda_qkv_prefill(
                        (q, k, v), (self.q_conv1d, self.k_conv1d, self.v_conv1d),
                        (cq0, ck0, cv0), enabled=True,
                    )
                    if prepared is not None:
                        preparation_mode = "qkv_silu"
                if prepared is None and self._fused_kda_prefill and seg_t > 1:
                    prepared = (
                        kda_conv_prefill(q, self.q_conv1d, cq0),
                        kda_conv_prefill(k, self.k_conv1d, ck0),
                        kda_conv_prefill(v, self.v_conv1d, cv0),
                    )
                if prepared is not None and all(item is not None for item in prepared):
                    (q, cq1), (k, ck1), (v, cv1) = prepared
                    if not self._fused_kda_prefill_observed:
                        _LOG.info(
                            "GLM KDA prefill convolution active: tokens=%d channels=%d mode=%s",
                            seg_t, H * K, preparation_mode,
                        )
                        self._fused_kda_prefill_observed = True
                else:
                    q, cq1 = short_conv(q, self.q_conv1d, cq0)
                    k, ck1 = short_conv(k, self.k_conv1d, ck0)
                    v, cv1 = short_conv(v, self.v_conv1d, cv0)
            q = l2norm(q.reshape(B, seg_t, H, K))
            k = l2norm(k.reshape(B, seg_t, H, K))
            v = v.reshape(B, seg_t, H, K)
            g = self._gate(seg)
            beta = mx.sigmoid(self.b_proj(seg).astype(mx.float32))
            if seg_t == 1 and s0 is not None:
                fused_step = glm5_kda_step_decode(
                    q[:, 0],
                    k[:, 0],
                    v[:, 0],
                    g[:, 0],
                    beta[:, 0],
                    s0,
                    enabled=self._fused_kda_step,
                )
                if fused_step is not None:
                    o, s1 = fused_step
                else:
                    o, s1 = kda_step(
                        q[:, 0],
                        k[:, 0],
                        v[:, 0],
                        g[:, 0],
                        beta[:, 0],
                        s0,
                    )
                o = o[:, None]
            elif seg_t <= 64:
                o, s1 = kda_recurrent(q, k, v, g, beta, s0)
            else:
                o, s1 = kda_chunked(q, k, v, g, beta, s0)
            gate = self.g_b_proj(self.g_a_proj(seg)).reshape(B, seg_t, H, K)
            gated = sigmoid_gated_rmsnorm_small_rows(
                o,
                gate,
                self.o_norm,
                self.rms_eps,
                output_dtype=seg.dtype,
                enabled=self._fused_gated_norm,
            )
            if gated is None:
                o32 = o.astype(mx.float32)
                o32 = o32 * mx.rsqrt(
                    mx.mean(o32 * o32, axis=-1, keepdims=True) + self.rms_eps
                )
                o32 = self.o_norm.astype(mx.float32) * o32
                gated = (o32 * mx.sigmoid(gate.astype(mx.float32))).astype(
                    seg.dtype
                )
            projected = self.o_proj(gated.reshape(B, seg_t, H * K))
            return projected, cq1, ck1, cv1, s1

        if (
            cache is not None
            and isinstance(cache, Glm5KDACache)
            and 0 < n_confirmed < T
        ):
            if self._vectorized_speculative_verify:
                # Compute every position-independent operation once over the
                # full verify slab.  Only the true KDA recurrence stays
                # sequential; it returns each accepted-prefix state for exact
                # partial rollback.  This preserves speculative correctness
                # without paying (draft_depth + 1) separate QMM/gate/output
                # projection dispatches in every KDA layer.
                q, k, v = self._project_qkv(x)
                q, conv_q, conv_q_states = short_conv_with_states(
                    q, self.q_conv1d, conv_q
                )
                k, conv_k, conv_k_states = short_conv_with_states(
                    k, self.k_conv1d, conv_k
                )
                v, conv_v, conv_v_states = short_conv_with_states(
                    v, self.v_conv1d, conv_v
                )
                q = l2norm(q.reshape(B, T, H, K))
                k = l2norm(k.reshape(B, T, H, K))
                v = v.reshape(B, T, H, K)
                g = self._gate(x)
                beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))
                o, state, recurrent_states = kda_recurrent_with_states(
                    q, k, v, g, beta, state
                )
                gate = self.g_b_proj(self.g_a_proj(x)).reshape(B, T, H, K)
                gated = sigmoid_gated_rmsnorm_small_rows(
                    o,
                    gate,
                    self.o_norm,
                    self.rms_eps,
                    output_dtype=x.dtype,
                    enabled=self._fused_gated_norm,
                )
                if gated is None:
                    o32 = o.astype(mx.float32)
                    o32 = o32 * mx.rsqrt(
                        mx.mean(o32 * o32, axis=-1, keepdims=True)
                        + self.rms_eps
                    )
                    o32 = self.o_norm.astype(mx.float32) * o32
                    gated = (
                        o32 * mx.sigmoid(gate.astype(mx.float32))
                    ).astype(x.dtype)
                result = self.o_proj(gated.reshape(B, T, H * K))
                cache.set_speculative_states(
                    list(
                        zip(
                            conv_q_states,
                            conv_k_states,
                            conv_v_states,
                            recurrent_states,
                        )
                    )
                )
            else:
                outputs = []
                states = []
                # The confirmed prefix may contain more than one token in
                # future callers. Each speculative token is advanced
                # separately so every rollback boundary has an exact state.
                out, conv_q, conv_k, conv_v, state = run_segment(
                    x[:, :n_confirmed], conv_q, conv_k, conv_v, state
                )
                outputs.append(out)
                states.append((conv_q, conv_k, conv_v, state))
                for pos in range(n_confirmed, T):
                    out, conv_q, conv_k, conv_v, state = run_segment(
                        x[:, pos : pos + 1], conv_q, conv_k, conv_v, state
                    )
                    outputs.append(out)
                    states.append((conv_q, conv_k, conv_v, state))
                cache.set_speculative_states(states)
                result = mx.concatenate(outputs, axis=1)
        else:
            result, conv_q, conv_k, conv_v, state = run_segment(
                x, conv_q, conv_k, conv_v, state
            )

        if cache is not None:
            cache.cache[KDA_CONV_Q] = conv_q
            cache.cache[KDA_CONV_K] = conv_k
            cache.cache[KDA_CONV_V] = conv_v
            cache.cache[KDA_STATE] = state
        return result


# ---------------------------------------------------------------- DSA ------
def _dsa_bf16_state_requested() -> bool:
    """BF16 DSA indexer state/scoring (VMLX_GLM5_DSA_BF16, default off).

    Halves the pooled-key cache footprint and scores pools in the packed
    cache's own 16-bit dtype instead of casting the full history to FP32
    every step. Internal layer-norm/gate math stays FP32; only the stored
    state and the score contraction narrow. Default-off until the retrieval
    gate (needle across the index_topk boundary) passes live per bundle.
    """
    value = os.environ.get("VMLX_GLM5_DSA_BF16", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


class Glm5NextIndexer(nn.Module):
    """DSA indexer with k-pool compression (port of Glm5NextTextIndexer).

    Scores softmax-compressed pools of ``index_kpool`` index keys with relu'd
    per-head dot products, weights heads via ``weights_proj / sqrt(H)``,
    selects the top ``index_topk // index_kpool`` pools whose FINAL token is
    causally visible to the query, expands them back to raw token indices,
    and always appends the query's incomplete tail pool as raw indices.

    Serving restriction: batch size 1, no padding (the vMLX single-active
    lane) — valid_keys is all-true and first_key is 0, which removes the
    padding-offset re-basing of the official implementation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.topk = args.index_topk
        self.kpool = args.index_kpool
        self.select_tail = args.index_kpool_always_select_tail
        self.scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.index_kpool_compress_ape = mx.zeros((self.kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, args.hidden_size))
        self._fused_score_decode = fused_sparse_index_score_requested(
            "glm5_next"
        )
        self._state_dtype = (
            mx.bfloat16 if _dsa_bf16_state_requested() else mx.float32
        )

    def packed_states(self, x: mx.array) -> mx.array:
        """Per-token indexer state: [B, T, head_dim(k) + head_dim(gate)]."""
        k = mx.fast.layer_norm(
            self.wk(x).astype(mx.float32),
            self.k_norm.weight.astype(mx.float32),
            self.k_norm.bias.astype(mx.float32),
            1e-6,
        )
        gate = x.astype(mx.float32) @ self.index_kpool_compress_gate.astype(mx.float32).T
        return mx.concatenate([k, gate], axis=-1).astype(self._state_dtype)

    def compress_pool_keys(self, packed: mx.array) -> mx.array:
        """Compress one or more complete raw k-pools exactly once."""

        B, T, width = packed.shape
        if T % self.kpool:
            raise ValueError("DSA pool compression requires complete pools")
        if width != 2 * self.head_dim:
            raise ValueError("DSA packed-state width differs from indexer contract")
        n_pools = T // self.kpool
        if n_pools == 0:
            return mx.zeros((B, 0, self.head_dim), dtype=self._state_dtype)
        keys, gates = mx.split(
            packed.astype(mx.float32), [self.head_dim], axis=-1
        )
        keys = keys.reshape(B, n_pools, self.kpool, self.head_dim)
        gates = gates.reshape(B, n_pools, self.kpool, self.head_dim)
        logits = gates + self.index_kpool_compress_ape.astype(mx.float32)[
            None, None
        ]
        return mx.sum(mx.softmax(logits, axis=2) * keys, axis=2).astype(
            self._state_dtype
        )

    def update_pool_cache(
        self,
        cache: Glm5MLACache,
        packed: mx.array,
    ) -> mx.array | None:
        """Append only newly completed DSA pools to the typed MLA cache."""

        complete = int(packed.shape[1]) // self.kpool
        existing = cache.pool_keys
        cached = 0 if existing is None else int(existing.shape[1])
        if cached > complete:
            raise ValueError("DSA pool cache is ahead of packed token history")
        if cached < complete:
            start = cached * self.kpool
            end = complete * self.kpool
            new_keys = self.compress_pool_keys(packed[:, start:end, :])
            existing = cache.update_pool_keys(new_keys)
        return existing

    def topk_indices(
        self,
        x: mx.array,
        q_resid: mx.array,
        packed: mx.array,
        q_positions: mx.array,
        pool_keys: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return (indices [B, S, W], valid [B, S, W]) raw token selections."""
        B, S = x.shape[:2]
        T = packed.shape[1]
        P = self.kpool
        n_pools = T // P  # complete pools only; the tail covers the remainder

        q = self.wq_b(q_resid).reshape(B, S, self.n_heads, self.head_dim).astype(
            self._state_dtype
        )
        head_w = (self.weights_proj(x).astype(self._state_dtype)
                  * (self.n_heads ** -0.5))                    # [B, S, H]

        if n_pools > 0:
            if pool_keys is None:
                pool_keys = self.compress_pool_keys(
                    packed[:, : n_pools * P, :]
                )
            elif tuple(pool_keys.shape) != (B, n_pools, self.head_dim):
                raise ValueError(
                    "DSA cached pool-key shape differs from packed history: "
                    f"cached={tuple(pool_keys.shape)} "
                    f"expected={(B, n_pools, self.head_dim)}"
                )

            # scores per idx head then head-weighted sum: [B, S, n_pools]
            pool_scores = sparse_index_scores_decode(
                q,
                pool_keys,
                family="glm5_next",
                scale=self.scale,
                head_weights=head_w,
                enabled=self._fused_score_decode,
            )
            if pool_scores is None:
                scores = mx.einsum("bshd,bpd->bshp", q, pool_keys)
                scores = mx.maximum(scores * self.scale, 0.0)
                pool_scores = mx.einsum("bsh,bshp->bsp", head_w, scores)

            # a pool is selectable only if its final token is visible
            pool_end = mx.arange(n_pools) * P + (P - 1)        # [n_pools]
            visible = pool_end[None, None, :] <= q_positions[None, :, None]
            pool_scores = mx.where(visible, pool_scores, mx.full(pool_scores.shape, -mx.inf))

            select_k = min(self.topk // P, n_pools)
            sel = mx.argpartition(-pool_scores, kth=select_k - 1, axis=-1)[..., :select_k]
            sel_visible = mx.take_along_axis(visible.astype(mx.bool_)
                                             if visible.ndim == 3 else visible,
                                             sel, axis=-1)
            # expand pools -> raw indices [B, S, select_k * P]
            raw = (sel[..., None] * P + mx.arange(P)[None, None, None, :])
            raw = raw.reshape(B, S, select_k * P)
            raw_valid = mx.repeat(sel_visible, P, axis=-1).reshape(B, S, select_k * P)
        else:
            raw = mx.zeros((B, S, 0), dtype=mx.int32)
            raw_valid = mx.zeros((B, S, 0), dtype=mx.bool_)

        if self.select_tail and P > 1:
            count = (q_positions + 1)                          # visible tokens per query
            tail_start = (count // P) * P                      # [S]
            offs = mx.arange(P - 1)
            tail = tail_start[:, None] + offs[None, :]         # [S, P-1]
            tail_valid = tail < count[:, None]
            tail = mx.broadcast_to(tail[None], (B, S, P - 1))
            tail_valid = mx.broadcast_to(tail_valid[None], (B, S, P - 1))
            raw = mx.concatenate([raw.astype(mx.int32), tail.astype(mx.int32)], axis=-1)
            raw_valid = mx.concatenate([raw_valid, tail_valid], axis=-1)

        return raw, raw_valid


# ---------------------------------------------------------------- MLA ------
class MLAAttention(nn.Module):
    """DeepSeek-V3 MLA, pure NoPE, with the DSA sparse path.

    Dense causal attention whenever the total sequence fits index_topk (the
    top-k would select every key — bit-exact, and the calibration/KL evidence
    all ran under this bound). Beyond index_topk, the DSA indexer selects
    each query's visible set; decode gathers the selected K/V rows, prefill
    builds an additive visibility mask.
    """

    gather_element_budget = 268_435_456

    def __init__(self, args: ModelArgs):
        super().__init__()
        d = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.qk = args.qk_nope_head_dim
        self.vd = args.v_head_dim
        self.index_topk = args.index_topk
        self.q_a_proj = nn.Linear(d, args.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.n_heads * self.qk, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, args.kv_lora_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(args.kv_lora_rank, self.n_heads * (self.qk + self.vd), bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.vd, d, bias=False)
        self.indexer = Glm5NextIndexer(args)
        self.scale = self.qk ** -0.5
        # Absorbed per-head kv_b factors, hydrated lazily after quantization.
        # They are derived runtime state rather than checkpoint parameters.
        self._w_kb_nope: mx.array | None = None
        self._w_kb_v: mx.array | None = None

    def _kb_factors(self) -> tuple[mx.array, mx.array]:
        """Return dequantized per-head K/V factors for absorbed MLA."""

        if self._w_kb_nope is None:
            w = self.kv_b_proj.weight
            quantized = hasattr(self.kv_b_proj, "scales")
            if quantized:
                w = mx.dequantize(
                    w,
                    self.kv_b_proj.scales,
                    getattr(self.kv_b_proj, "biases", None),
                    group_size=self.kv_b_proj.group_size,
                    bits=self.kv_b_proj.bits,
                    mode=getattr(self.kv_b_proj, "mode", "affine"),
                )
            w = w.reshape(
                self.n_heads,
                self.qk + self.vd,
                -1,
            ).astype(mx.bfloat16 if quantized else w.dtype)
            self._w_kb_nope = w[:, : self.qk, :]
            self._w_kb_v = w[:, self.qk :, :]
            mx.eval(self._w_kb_nope, self._w_kb_v)
        return self._w_kb_nope, self._w_kb_v

    def _gather_absorbed_attention(
        self,
        queries: mx.array,
        latent: mx.array,
        indices: mx.array,
        valid: mx.array,
        *,
        past: int,
    ) -> mx.array:
        """Attend to each query's selected latent rows without dense masks."""

        B, n_heads, S, rank = queries.shape
        K = int(indices.shape[-1])
        flat_latent = latent[:, 0]
        total = int(flat_latent.shape[1])
        tile = max(
            1,
            min(S, self.gather_element_budget // max(K * rank, 1)),
        )
        outputs = []
        for start in range(0, S, tile):
            stop = min(start + tile, S)
            rows = stop - start
            idx = indices[:, start:stop]
            valid_rows = valid[:, start:stop]
            safe = mx.where(valid_rows, idx, mx.zeros_like(idx))
            if B == 1:
                gathered = mx.take(
                    flat_latent[0], safe.reshape(rows * K), axis=0
                )
            else:
                shifted = (
                    safe.reshape(B, rows * K)
                    + (mx.arange(B, dtype=safe.dtype) * total)[:, None]
                ).reshape(B * rows * K)
                gathered = mx.take(
                    flat_latent.reshape(B * total, rank), shifted, axis=0
                )
            keys = gathered.reshape(B * rows, 1, K, rank)
            q = (
                queries[:, :, start:stop]
                .transpose(0, 2, 1, 3)
                .reshape(B * rows, 1, n_heads, rank)
            )
            q_pos = (past + mx.arange(start, stop))[None, :, None]
            allowed = valid_rows & (idx <= q_pos)
            bias = mx.where(
                allowed,
                mx.array(0.0, dtype=mx.float32),
                mx.array(-mx.inf, dtype=mx.float32),
            ).reshape(B * rows, 1, 1, K)
            if S == 1:
                attended = mx.fast.scaled_dot_product_attention(
                    q.astype(mx.float32),
                    keys.astype(mx.float32),
                    keys.astype(mx.float32),
                    scale=self.scale,
                    mask=bias,
                ).astype(queries.dtype)
            else:
                attended = mx.fast.scaled_dot_product_attention(
                    q,
                    keys,
                    keys,
                    scale=self.scale,
                    mask=bias.astype(queries.dtype),
                )
            outputs.append(
                attended.reshape(B, rows, n_heads, rank).transpose(0, 2, 1, 3)
            )
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)

    def _materialized_from_latent_attention(
        self,
        q: mx.array,
        latent: mx.array,
    ) -> mx.array:
        """Use compact persisted state with the faster dense-prefill algebra.

        Absorption makes decode cheaper, but for a multi-token dense prefill it
        widens both attention operands from qk/v head dimensions to the latent
        rank.  Re-expanding the fetched latent once keeps the persisted/SSD
        representation compact while retaining the existing prefill kernel.
        """

        B, _, T, _ = q.shape
        total = int(latent.shape[2])
        kv = self.kv_b_proj(latent[:, 0])
        kv = kv.reshape(
            B,
            total,
            self.n_heads,
            self.qk + self.vd,
        ).transpose(0, 2, 1, 3)
        keys, values = mx.split(kv, [self.qk], axis=-1)
        attended = mx.fast.scaled_dot_product_attention(
            q,
            keys,
            values,
            scale=self.scale,
            mask="causal",
        )
        values = attended.transpose(0, 2, 1, 3).reshape(
            B, T, self.n_heads * self.vd
        )
        return self.o_proj(values)

    @staticmethod
    def _log_mla_path_once(path: str, *, query_tokens: int, total: int) -> None:
        if path in _GLM5_MLA_PATHS_LOGGED:
            return
        _GLM5_MLA_PATHS_LOGGED.add(path)
        _LOG.info(
            "glm5_next MLA path: %s (query_tokens=%d total_tokens=%d)",
            path,
            query_tokens,
            total,
        )

    def _absorbed_attention(
        self,
        x: mx.array,
        q: mx.array,
        q_resid: mx.array,
        latent_now: mx.array,
        packed_now: mx.array,
        cache: Glm5MLACache | None,
    ) -> mx.array:
        """Run pure-NoPE MLA against the shared latent cache."""

        B, T, _ = x.shape
        past = cache.offset if cache is not None else 0
        latent = latent_now[:, None]
        packed = packed_now
        if cache is not None:
            latent = cache.update_latent(latent)
            packed = cache.update_packed(packed)

        total = int(latent.shape[2])
        if T > 1 and total <= self.index_topk:
            self._log_mla_path_once(
                "compact-cache/materialized-dense-prefill",
                query_tokens=T,
                total=total,
            )
            return self._materialized_from_latent_attention(q, latent)

        w_k, w_v = self._kb_factors()
        q_eff = mx.matmul(q, w_k[None].astype(q.dtype))
        if total <= self.index_topk:
            self._log_mla_path_once(
                "absorbed-dense-decode",
                query_tokens=T,
                total=total,
            )
            if T == 1:
                attended = mx.fast.scaled_dot_product_attention(
                    q_eff.astype(mx.float32),
                    latent.astype(mx.float32),
                    latent.astype(mx.float32),
                    scale=self.scale,
                ).astype(q.dtype)
            else:
                attended = mx.fast.scaled_dot_product_attention(
                    q_eff,
                    latent,
                    latent,
                    scale=self.scale,
                    mask="causal",
                )
        else:
            self._log_mla_path_once(
                "absorbed-sparse-dsa",
                query_tokens=T,
                total=total,
            )
            q_positions = past + mx.arange(T)
            pool_keys = (
                self.indexer.update_pool_cache(cache, packed)
                if cache is not None
                else None
            )
            indices, valid = self.indexer.topk_indices(
                x,
                q_resid,
                packed,
                q_positions,
                pool_keys=pool_keys,
            )
            attended = self._gather_absorbed_attention(
                q_eff,
                latent,
                indices,
                valid,
                past=past,
            )
        values = mx.matmul(
            attended,
            w_v[None].swapaxes(-1, -2).astype(attended.dtype),
        )
        values = values.transpose(0, 2, 1, 3).reshape(
            B, T, self.n_heads * self.vd
        )
        return self.o_proj(values)

    def __call__(self, x: mx.array, cache: Optional[Glm5MLACache] = None):
        B, T, _ = x.shape
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid)
        q = q.reshape(B, T, self.n_heads, self.qk).transpose(0, 2, 1, 3)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        packed = self.indexer.packed_states(x)
        absorbed = (
            cache.absorbed
            if isinstance(cache, Glm5MLACache)
            else glm5_mla_absorb_enabled()
        )
        if absorbed:
            return self._absorbed_attention(
                x,
                q,
                q_resid,
                latent,
                packed,
                cache,
            )

        kv = self.kv_b_proj(latent)
        kv = kv.reshape(B, T, self.n_heads, self.qk + self.vd).transpose(0, 2, 1, 3)
        k, v = mx.split(kv, [self.qk], axis=-1)

        offset = 0
        if cache is not None:
            offset = cache.offset
            k, v = cache.update_kv(k, v)
            packed = cache.update_packed(packed)

        total = offset + T
        q32 = q.astype(mx.float32)
        k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32)

        if total <= self.index_topk:
            # Dense bypass: top-k selects every key — bit-exact.
            o = mx.fast.scaled_dot_product_attention(
                q32, k32, v32, scale=self.scale,
                mask="causal" if T > 1 else None)
        else:
            if B != 1:
                raise ValueError(
                    "glm5_next sparse DSA path currently supports batch "
                    "size 1 (single-active serving lane)")
            q_positions = offset + mx.arange(T)
            pool_keys = (
                self.indexer.update_pool_cache(cache, packed)
                if cache is not None
                else None
            )
            idx, idx_valid = self.indexer.topk_indices(
                x,
                q_resid,
                packed,
                q_positions,
                pool_keys=pool_keys,
            )
            if T == 1:
                # Decode: gather selected K/V rows, dense SDPA over them.
                flat = idx[0, 0]
                valid = idx_valid[0, 0]
                flat = mx.where(valid, flat, mx.zeros_like(flat))
                # dedup not required: duplicate keys receive identical
                # scores; softmax over a multiset changes the result, so
                # invalid slots are pointed at index 0 and masked instead.
                gk = mx.take(k32, flat, axis=2)                # [B, H, W, D]
                gv = mx.take(v32, flat, axis=2)
                bias = mx.where(valid, mx.zeros(valid.shape, dtype=mx.float32),
                                mx.full(valid.shape, -1e30))
                o = mx.fast.scaled_dot_product_attention(
                    q32, gk, gv, scale=self.scale,
                    mask=bias[None, None, None, :])
            else:
                # Prefill chunk: additive visibility mask over the full
                # width, built via a trash-column scatter (W duplicates are
                # idempotent for set-to-visible).
                W = idx.shape[-1]
                safe = mx.where(idx_valid, idx, mx.full(idx.shape, total, dtype=idx.dtype))
                mask = mx.zeros((B, T, total + 1), dtype=mx.bool_)
                mask = mx.put_along_axis(
                    mask, safe.astype(mx.int64),
                    mx.ones(safe.shape, dtype=mx.bool_), axis=-1)
                mask = mask[..., :total]
                bias = mx.where(mask, mx.zeros(mask.shape, dtype=mx.float32),
                                mx.full(mask.shape, -1e30))
                o = mx.fast.scaled_dot_product_attention(
                    q32, k32, v32, scale=self.scale,
                    mask=bias[:, None])
        o = o.astype(x.dtype).transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.vd)
        return self.o_proj(o)


# ---------------------------------------------------------------- MoE ------
def _clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class DenseMLP(nn.Module):
    def __init__(self, args: ModelArgs, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, args.hidden_size, bias=False)
        self.limit = args.swiglu_limit
        self.gate_up_group = None

    def prepare_runtime(self) -> bool:
        """Group compatible packed gate/up rows without retaining a copy."""

        linears = (self.gate_proj, self.up_proj)
        if quantized_projection_group_reason(linears) is not None:
            return False
        group = QuantizedProjectionGroup(linears)
        mx.eval(group.weight, group.scales, group.biases)
        self.gate_up_group = group
        self.gate_proj = self.up_proj = None
        return True

    def __call__(self, x):
        if self.gate_up_group is not None:
            gate, up = self.gate_up_group(x)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(_clamped_swiglu(gate, up, self.limit))


class ClampedSwiGLU(nn.Module):
    """SwitchGLU activation hook: silu(clamp(gate)) * clamp(up), limit ±10."""

    def __init__(self, limit: float):
        super().__init__()
        self._limit = limit

    def __call__(self, x_up, x_gate):
        return _clamped_swiglu(x_gate, x_up, self._limit)


class MoEBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        E = args.n_routed_experts
        self.k = args.num_experts_per_tok
        self.norm_topk = args.norm_topk_prob
        self.scaling = args.routed_scaling_factor
        self.gate = nn.Linear(args.hidden_size, E, bias=False)
        self.e_score_correction_bias = mx.zeros((E,))
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, E,
                                    activation=ClampedSwiGLU(args.swiglu_limit))
        self.shared_experts = DenseMLP(args, args.moe_intermediate_size * args.n_shared_experts)

    def __call__(self, x: mx.array):
        # Router in fp32 (weights are fp32 keeps; logits/topk fp32 by contract).
        logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        idx = mx.argpartition(-choice, kth=self.k - 1, axis=-1)[..., : self.k]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w * self.scaling
        routed, pair_fused = affine_moe_routed_output(
            self.switch_mlp, x, idx, w
        )
        if not pair_fused:
            routed = self.switch_mlp(x, idx)               # [B, T, k, d]
            routed = mx.sum(
                routed * w[..., None].astype(routed.dtype), axis=-2
            )
        return routed.astype(x.dtype) + self.shared_experts(x)


# ---------------------------------------------------------------- layers ---
class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        self.self_attn = KDAAttention(args) if self.is_linear else MLAAttention(args)
        self.mlp = (DenseMLP(args, args.intermediate_size)
                    if layer_idx < args.first_k_dense_replace else MoEBlock(args))
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)
        self._fused_hc_place = fused_glm5_hc_place_requested()

    def __call__(self, streams: mx.array, cache=None, n_confirmed: int = 0):
        residual = streams
        post, comb, x = self.attn_hc(streams)
        attn_input = self.input_layernorm(x)
        if self.is_linear:
            x = self.self_attn(
                attn_input, cache=cache, n_confirmed=n_confirmed
            )
        else:
            x = self.self_attn(attn_input, cache=cache)
        streams = hc_place(
            post, comb, x, residual, fused_decode=self._fused_hc_place
        )

        residual = streams
        post, comb, x = self.ffn_hc(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        return hc_place(
            post, comb, x, residual, fused_decode=self._fused_hc_place
        )


class Glm5NextMTPSharedHead(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)


class Glm5NextMTP(nn.Module):
    """Checkpoint-native layer-45 draft head.

    The checkpoint does not store mHC parameters for this layer.  Its block
    is the ordinary residual form used by the upstream MTP implementation:
    MLA residual followed by MoE residual, then a private output-head norm.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.enorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.hnorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = MLAAttention(args)
        self.post_attention_layernorm = RMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.mlp = MoEBlock(args)
        self.shared_head = Glm5NextMTPSharedHead(args)

    def __call__(self, previous_hidden, next_token_ids, embed_tokens, cache=None):
        embedding = embed_tokens(next_token_ids)
        fused = self.eh_proj(
            mx.concatenate(
                [self.enorm(embedding), self.hnorm(previous_hidden)], axis=-1
            )
        )
        residual = fused
        fused = residual + self.self_attn(
            self.input_layernorm(fused), cache=cache
        )
        residual = fused
        return residual + self.mlp(self.post_attention_layernorm(fused))


class Glm5NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        cache=None,
        n_confirmed: int = 0,
        return_pre_norm: bool = False,
        inputs_embeds: mx.array | None = None,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("glm5_next requires input_ids or inputs_embeds")
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        offset = 0
        if cache is not None:
            for c in cache:
                if isinstance(c, (Glm5MLACache, KVCache)):
                    offset = c.offset
                    break
        seen = offset + x.shape[1]
        if seen > self.args.max_position_embeddings:
            raise ValueError(
                f"glm5_next context limit is {self.args.max_position_embeddings} "
                f"tokens (requested {seen})")
        streams = mx.broadcast_to(x[:, :, None, :],
                                  (*x.shape[:2], self.args.hc_mult, x.shape[-1]))
        for i, layer in enumerate(self.layers):
            lc = cache[i] if cache is not None else None
            streams = layer(
                streams, cache=lc, n_confirmed=n_confirmed
            )
        hidden = mx.mean(streams, axis=2)
        return hidden if return_pre_norm else self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Scheduler runtime policy detects the family from model.config
        # (dict) — provide it so glm5_next-specific policy (e.g. the
        # fail-closed prefix-cache gate) reliably arms.
        self.config = {"model_type": args.model_type}
        self.model = Glm5NextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        from vmlx_engine.patches.mlx_lm_mtp import is_mtp_active

        if args.num_nextn_predict_layers > 0 and is_mtp_active():
            self.mtp = Glm5NextMTP(args)

    def __call__(
        self,
        inputs: mx.array | None = None,
        cache=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        n_confirmed: int = 0,
        inputs_embeds: mx.array | None = None,
        **kwargs,
    ):
        hidden = self.model(
            inputs,
            cache=cache,
            n_confirmed=n_confirmed,
            return_pre_norm=True,
            inputs_embeds=inputs_embeds,
        )
        if not return_hidden:
            from vmlx_engine.native_mtp_prompt_priming import (
                capture_prefill,
                capture_requested,
            )

            # The BatchGenerator request plan is the sole activation gate.
            # Caching an env-derived boolean at model construction made an
            # otherwise armed request silently miss prompt capture when the
            # runtime policy was resolved later in the load/generation
            # lifecycle.  No plan means the ordinary forward stays unchanged.
            if capture_requested(self):
                capture_prefill(self, inputs, hidden, cache)
        if not return_logits:
            return hidden
        logits = self.lm_head(self.model.norm(hidden))
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
    ):
        hidden = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache[0] if mtp_cache else None,
        )
        logits = self.lm_head(self.mtp.shared_head.norm(hidden))
        return (logits, hidden) if return_hidden else logits

    def make_mtp_cache(self):
        return (
            [Glm5MLACache(self.args.index_kpool)]
            if hasattr(self, "mtp")
            else []
        )

    def prepare_acceleration(self) -> dict[str, int]:
        """Install exact launch-reduction groups after checkpoint hydration.

        Metadata dtype correctness (FP16 affine scales/biases vs BF16
        activations promoting every quantized matmul to FP32) is owned by the
        load boundary: the JANG loaders resolve the compute dtype and
        ``mlx_memory.maybe_harmonize_quant_metadata_dtypes`` runs on both
        production routes before this hook, so grouping and fused-MoE
        registration below already observe the final dtypes.
        """

        fused_moe_pair_modules = install_affine_moe_pair_decode(
            self, family="glm5_next"
        )

        base_kda_groups = 0
        base_dense_gate_up_groups = 0
        for layer in self.model.layers:
            if layer.is_linear and layer.self_attn.prepare_runtime():
                base_kda_groups += 1
            dense = (
                layer.mlp
                if isinstance(layer.mlp, DenseMLP)
                else layer.mlp.shared_experts
            )
            if dense.prepare_runtime():
                base_dense_gate_up_groups += 1
        mtp_dense_gate_up_groups = 0
        if hasattr(self, "mtp") and self.mtp.mlp.shared_experts.prepare_runtime():
            mtp_dense_gate_up_groups = 1
        mx.clear_cache()
        return {
            "base_kda_qkv_groups": base_kda_groups,
            "base_dense_gate_up_groups": base_dense_gate_up_groups,
            "mtp_dense_gate_up_groups": mtp_dense_gate_up_groups,
            "vectorized_kda_verify_layers": sum(
                int(
                    layer.is_linear
                    and layer.self_attn._vectorized_speculative_verify
                )
                for layer in self.model.layers
            ),
            "fused_moe_pair_modules": fused_moe_pair_modules,
            "base_launches_removed_per_forward": (
                2 * base_kda_groups + base_dense_gate_up_groups
            ),
            "mtp_launches_removed_per_forward": mtp_dense_gate_up_groups,
        }

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.model.layers:
            if layer.is_linear:
                caches.append(Glm5KDACache())
            else:
                caches.append(Glm5MLACache(self.args.index_kpool))
        return caches

    def sanitize(self, weights: dict) -> dict:
        """Drop subsystems the text runtime does not construct.

        The JANG bundle stores runtime naming already (attn_hc/ffn_hc,
        stacked switch_mlp experts, squeezed [C,W] convs, bare o_norm,
        indexer under self_attn.indexer); only the vision tower and the MTP
        block (layer == num_hidden_layers) are removed. Everything else must
        match the module tree exactly (strict load).
        """
        n = self.args.num_hidden_layers
        mtp_prefix = f"model.layers.{n}."
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(("visual.", "model.visual.")):
                continue
            if key.startswith(mtp_prefix):
                if hasattr(self, "mtp"):
                    sanitized[f"mtp.{key[len(mtp_prefix):]}"] = value
                continue
            sanitized[key] = value
        return sanitized

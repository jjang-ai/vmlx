# SPDX-License-Identifier: Apache-2.0
"""
Affine MoE decode fast-path — fused dequant+matvec Metal kernels.

Replaces stock SwitchGLU gather_qmm for batch=1 decode on affine-quantized
JANG models (2-bit and 3-bit weights with per-group scale/bias).  The stock
path achieves ~2% of peak bandwidth at k=6 because each expert matvec is too
small to saturate the GPU.  These kernels fuse gather + dequant + dot-product
into a single pass with no intermediate materialization.

Architecture: two-pass split reduction.
  Pass 1 (Metal): each thread computes a partial dot-product over a chunk of
    groups.  Grid = (k, out_dim, n_chunks).  Output: fp32 partial sums
    [k, out_dim, n_chunks].
  Pass 2 (host): mx.sum across chunks → [k, out_dim] fp16.

  Splitting the group loop across chunks increases GPU occupancy.
  n_chunks=2, threadgroup=(1, 128, 1) is optimal on M4 Max.

Measured: 4.9 → 17.1 tok/s (3.46×) on DeepSeek-V4-Flash JANG, M4 Max 128 GB.

Usage::

    from vmlx_engine.metal.affine_moe_decode import install_affine_moe_fastpath
    install_affine_moe_fastpath(model)

The patch is scoped to batch=1 decode (indices.size < 64).  Prefill and
large-batch calls fall through to the original SwitchGLU.__call__.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Kernel source strings — must stay in sync with affine_moe_decode.metal
# ---------------------------------------------------------------------------

_KERNEL_B2_G64 = """
    uint expert_local = thread_position_in_grid.x;
    uint out_d = thread_position_in_grid.y;
    uint chunk_id = thread_position_in_grid.z;
    if (expert_local >= (uint)k || out_d >= (uint)out_dim) return;
    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = (uint)n_groups * 4u;
    uint w_row_base = expert_id * (uint)out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * (uint)out_dim * (uint)n_groups + out_d * (uint)n_groups;
    uint groups_per_chunk = (uint)n_groups / (uint)n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;
    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = g * 64u;
        uint w_base = w_row_base + g * 4u;
        for (uint w = 0; w < 4u; w++) {
            uint packed = weights[w_base + w];
            uint x_off = x_base + w * 16u;
            for (uint i = 0; i < 16u; i++) {
                uint code = (packed >> (i * 2u)) & 0x3u;
                sum += (float)x[x_off + i] * ((float)code * scale + bias);
            }
        }
    }
    partial[expert_local * (uint)out_dim * (uint)n_chunks + out_d * (uint)n_chunks + chunk_id] = sum;
"""

_KERNEL_B3_G64 = """
    uint expert_local = thread_position_in_grid.x;
    uint out_d = thread_position_in_grid.y;
    uint chunk_id = thread_position_in_grid.z;
    if (expert_local >= (uint)k || out_d >= (uint)out_dim) return;
    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = (uint)n_groups * 6u;
    uint w_row_base = expert_id * (uint)out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * (uint)out_dim * (uint)n_groups + out_d * (uint)n_groups;
    uint groups_per_chunk = (uint)n_groups / (uint)n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;
    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = g * 64u;
        uint w_base = w_row_base + g * 6u;
        for (uint i = 0; i < 64u; i++) {
            uint bit_start = i * 3u;
            uint uint_idx = bit_start / 32u;
            uint bit_off = bit_start % 32u;
            uint packed = weights[w_base + uint_idx];
            uint code;
            if (bit_off <= 29u) { code = (packed >> bit_off) & 0x7u; }
            else {
                uint lo = packed >> bit_off;
                uint hi = weights[w_base + uint_idx + 1u];
                code = (lo | (hi << (32u - bit_off))) & 0x7u;
            }
            sum += (float)x[x_base + i] * ((float)code * scale + bias);
        }
    }
    partial[expert_local * (uint)out_dim * (uint)n_chunks + out_d * (uint)n_chunks + chunk_id] = sum;
"""

_KERNEL_B2_G32 = """
    uint expert_local = thread_position_in_grid.x;
    uint out_d = thread_position_in_grid.y;
    uint chunk_id = thread_position_in_grid.z;
    if (expert_local >= (uint)k || out_d >= (uint)out_dim) return;
    uint expert_id = (uint)expert_ids[expert_local];
    uint packed_per_row = (uint)n_groups * 2u;
    uint w_row_base = expert_id * (uint)out_dim * packed_per_row + out_d * packed_per_row;
    uint s_row_base = expert_id * (uint)out_dim * (uint)n_groups + out_d * (uint)n_groups;
    uint x_row_base = expert_local * (uint)hidden;
    uint groups_per_chunk = (uint)n_groups / (uint)n_chunks;
    uint g_start = chunk_id * groups_per_chunk;
    uint g_end = g_start + groups_per_chunk;
    float sum = 0.0f;
    for (uint g = g_start; g < g_end; g++) {
        float scale = (float)scales[s_row_base + g];
        float bias = (float)biases[s_row_base + g];
        uint x_base = x_row_base + g * 32u;
        uint w_base = w_row_base + g * 2u;
        for (uint w = 0; w < 2u; w++) {
            uint packed = weights[w_base + w];
            uint x_off = x_base + w * 16u;
            for (uint i = 0; i < 16u; i++) {
                uint code = (packed >> (i * 2u)) & 0x3u;
                sum += (float)x[x_off + i] * ((float)code * scale + bias);
            }
        }
    }
    partial[expert_local * (uint)out_dim * (uint)n_chunks + out_d * (uint)n_chunks + chunk_id] = sum;
"""

# ---------------------------------------------------------------------------
# Tunables — M4 Max optimal
# ---------------------------------------------------------------------------
N_CHUNKS = 2
THREADGROUP_Y = 128


class AffineMoEDecodeManager:
    """Compiles and caches affine MoE decode kernels.

    One instance per process is sufficient; kernels are compiled lazily
    on first use and cached for the lifetime of the manager.
    """

    def __init__(self, n_chunks: int = N_CHUNKS, threadgroup_y: int = THREADGROUP_Y):
        self.n_chunks = n_chunks
        self.threadgroup_y = threadgroup_y
        self._kernels: dict[str, Any] = {}
        self._compiled = False

    def _compile(self) -> None:
        if self._compiled:
            return
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is not available")
        specs = [
            ("b2_g64", _KERNEL_B2_G64),
            ("b3_g64", _KERNEL_B3_G64),
            ("b2_g32", _KERNEL_B2_G32),
        ]
        for name, src in specs:
            self._kernels[name] = mx.fast.metal_kernel(
                name=f"affine_moe_decode_{name}",
                input_names=[
                    "x", "weights", "scales", "biases", "expert_ids",
                    "hidden", "out_dim", "n_groups", "k", "n_chunks",
                ],
                output_names=["partial"],
                source=src,
                ensure_row_contiguous=False,
            )
        self._compiled = True
        logger.info(
            "Affine MoE decode kernels compiled (n_chunks=%d, tg_y=%d)",
            self.n_chunks, self.threadgroup_y,
        )

    def kernel_for(self, bits: int, group_size: int) -> Any:
        """Return the compiled kernel matching the given quantization params."""
        self._compile()
        key = f"b{bits}_g{group_size}"
        if key not in self._kernels:
            raise ValueError(
                f"No affine decode kernel for bits={bits}, group_size={group_size}. "
                f"Available: {sorted(self._kernels)}"
            )
        return self._kernels[key]

    def run_projection(
        self,
        x: "mx.array",
        weight: "mx.array",
        scales: "mx.array",
        biases: "mx.array",
        expert_ids_u16: "mx.array",
        hidden: int,
        out_dim: int,
        n_groups: int,
        k: int,
        bits: int,
        group_size: int,
    ) -> "mx.array":
        """Run one affine projection through the fused kernel.

        Returns fp16 output of shape [k, out_dim].
        """
        kern = self.kernel_for(bits, group_size)
        nc = self.n_chunks
        partial = kern(
            inputs=[
                x, weight, scales, biases, expert_ids_u16,
                mx.array(hidden, dtype=mx.uint32),
                mx.array(out_dim, dtype=mx.uint32),
                mx.array(n_groups, dtype=mx.uint32),
                mx.array(k, dtype=mx.uint32),
                mx.array(nc, dtype=mx.uint32),
            ],
            output_shapes=[(k, out_dim, nc)],
            output_dtypes=[mx.float32],
            grid=(k, out_dim, nc),
            threadgroup=(1, self.threadgroup_y, 1),
        )[0]
        return mx.sum(partial, axis=-1).astype(mx.float16)

    @property
    def is_available(self) -> bool:
        return _MLX_AVAILABLE


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_manager: AffineMoEDecodeManager | None = None


def get_manager() -> AffineMoEDecodeManager:
    global _manager
    if _manager is None:
        _manager = AffineMoEDecodeManager()
    return _manager


# ---------------------------------------------------------------------------
# SwitchGLU patch
# ---------------------------------------------------------------------------

def install_affine_moe_fastpath(model: Any) -> int:
    """Monkey-patch SwitchGLU.__call__ for affine-quantized decode.

    Only activates for batch=1 decode (indices.size < 64).  Prefill and
    large-batch calls fall through to the original implementation.

    Returns the number of SwitchGLU modules tagged for the fast path.
    """
    if not _MLX_AVAILABLE:
        logger.warning("MLX not available; affine MoE fast-path skipped")
        return 0

    from mlx_lm.models.switch_layers import SwitchGLU

    mgr = get_manager()
    _orig_call = getattr(SwitchGLU, "_affine_original_call", SwitchGLU.__call__)
    setattr(SwitchGLU, "_affine_original_call", _orig_call)

    # Per-layer constant cache (avoids mx.array allocation per token)
    _layer_cache: dict[int, dict] = {}

    def _get_layer_cfg(switch: Any) -> dict:
        sid = id(switch)
        if sid in _layer_cache:
            return _layer_cache[sid]
        gp, up, dp = switch.gate_proj, switch.up_proj, switch.down_proj
        cfg = {
            "gate": {
                "w": gp.weight, "s": gp.scales,
                "b": gp.biases if gp.biases is not None else mx.zeros_like(gp.scales),
                "out": gp.output_dims, "ng": gp.scales.shape[2],
                "bits": gp.bits, "gs": gp.group_size,
            },
            "up": {
                "w": up.weight, "s": up.scales,
                "b": up.biases if up.biases is not None else mx.zeros_like(up.scales),
                "out": up.output_dims, "ng": up.scales.shape[2],
                "bits": up.bits, "gs": up.group_size,
            },
            "down": {
                "w": dp.weight, "s": dp.scales,
                "b": dp.biases if dp.biases is not None else mx.zeros_like(dp.scales),
                "out": dp.output_dims, "ng": dp.scales.shape[2],
                "inter": up.output_dims,
                "bits": dp.bits, "gs": dp.group_size,
            },
            "hidden_u32": None,
        }
        _layer_cache[sid] = cfg
        return cfg

    def _affine_switchglu_call(self: Any, x: "mx.array", indices: "mx.array") -> "mx.array":
        batch = x.shape[0] * (x.shape[1] if x.ndim > 2 else 1)
        if batch > 1 or indices.size >= 64:
            return _orig_call(self, x, indices)

        idx = indices.reshape(-1)
        x1 = x.reshape(-1)
        k = idx.shape[0]
        idx16 = idx.astype(mx.uint16)
        c = _get_layer_cfg(self)

        if c["hidden_u32"] is None:
            c["hidden_u32"] = mx.array(x1.shape[0], dtype=mx.uint32)
        hidden_val = c["hidden_u32"]

        def _run(proj: dict, x_in: "mx.array", h_val: "mx.array") -> "mx.array":
            return mgr.run_projection(
                x_in, proj["w"], proj["s"], proj["b"], idx16,
                h_val.item() if hasattr(h_val, "item") else int(h_val),
                proj["out"], proj["ng"], k,
                proj["bits"], proj["gs"],
            )

        g = _run(c["gate"], x1, hidden_val)
        u = _run(c["up"], x1, hidden_val)
        act = self.activation(u, g)
        d = _run(c["down"], act, mx.array(c["down"]["inter"], dtype=mx.uint32))

        if x.ndim == 3:
            return d.reshape(x.shape[0], k, -1)
        return d.reshape(1, k, -1)

    SwitchGLU.__call__ = _affine_switchglu_call

    # Tag affine-quantized modules
    n_patched = 0
    for _, m in model.named_modules():
        if isinstance(m, SwitchGLU):
            gp = getattr(m, "gate_proj", None)
            if gp is not None and hasattr(gp, "bits") and hasattr(gp, "group_size"):
                n_patched += 1

    logger.info(
        "Affine MoE decode fast-path installed (%d SwitchGLU modules, "
        "n_chunks=%d, tg_y=%d)",
        n_patched, mgr.n_chunks, mgr.threadgroup_y,
    )
    return n_patched

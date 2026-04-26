"""
Codebook Kernel Manager - Fused kernels for codebook VQ MoE inference.

Uses MLX's mx.fast.metal_kernel API for Apple Silicon GPU kernels.

Kernels:
- codebook_moe_matvec: 3D indices [n_experts, out_dim, n_groups], selected top-k experts
- codebook_matvec_2d: 2D indices [out_dim, n_groups] for simple codebook VQ
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    logger.warning("MLX not available")


class CodebookKernelManager:
    """
    Manages fused Metal kernels for codebook VQ operations.

    Kernels:
    - codebook_moe_matvec: 3D indices, MoE top-k expert selection
    - codebook_matvec_2d: 2D indices, simple codebook VQ
    """

    def __init__(
        self,
        threads_per_threadgroup: int = 256,
        max_batch_size: int = 32,
    ):
        self._threads_per_threadgroup = threads_per_threadgroup
        self._max_batch_size = max_batch_size
        self._kernels: dict = {}
        self._metal_available = False

        if _MLX_AVAILABLE:
            self._initialize_metal()

    def _initialize_metal(self):
        """Initialize Metal and compile kernels."""
        try:
            if mx.metal.is_available():
                logger.info(f"Metal available, compiling kernels")
                self._metal_available = True
                self._compile_kernels()
            else:
                logger.warning("Metal not available on this device")
                self._metal_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Metal: {e}")
            self._metal_available = False

    def _compile_kernels(self):
        """Compile Metal kernels using mx.fast.metal_kernel."""
        try:
            self._kernels["codebook_moe_matvec"] = self._create_moe_kernel()
            self._kernels["codebook_matvec_2d"] = self._create_2d_kernel()
            logger.info("All kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile kernels: {e}")
            import traceback

            traceback.print_exc()
            self._kernels = {}

    def _create_moe_kernel(self):
        """Create the MoE codebook matvec kernel.

        Grid: (batch_size, k, out_dim)
        Each thread computes one output element out[b, expert, out_dim]
        """
        source = """
            uint b = thread_position_in_grid.x;
            uint expert_local = thread_position_in_grid.y;
            uint od = thread_position_in_grid.z;

            uint expert_id = (uint)expert_ids[expert_local];

            float sum = 0.0f;

            for (uint g = 0; g < (uint)n_groups; g++) {
                uint idx = (uint)indices[(expert_id * (uint)out_dim + od) * (uint)n_groups + g];
                uint cb_base = idx * (uint)group_size;
                uint x_base = b * (uint)in_dim + g * (uint)group_size;

                for (uint i = 0; i < (uint)group_size; i++) {
                    sum += (float)x[x_base + i] * (float)codebook[cb_base + i];
                }
            }

            uint out_idx = (b * (uint)k + expert_local) * (uint)out_dim + od;
            result[out_idx] = sum;
        """

        kernel = mx.fast.metal_kernel(
            name="codebook_moe_matvec",
            input_names=[
                "x",
                "codebook",
                "indices",
                "expert_ids",
                "batch_size",
                "k",
                "out_dim",
                "n_groups",
                "group_size",
                "in_dim",
            ],
            output_names=["result"],
            source=source,
            ensure_row_contiguous=True,
        )
        return kernel

    def _create_2d_kernel(self):
        """Create the 2D codebook matvec kernel.

        Grid: (batch_size, out_dim, 1)
        Each thread computes one output element out[b, od]
        """
        source = """
            uint b = thread_position_in_grid.x;
            uint od = thread_position_in_grid.y;

            float sum = 0.0f;

            for (uint g = 0; g < (uint)n_groups; g++) {
                uint idx = (uint)indices[od * (uint)n_groups + g];
                uint cb_base = idx * (uint)group_size;
                uint x_base = b * (uint)in_dim + g * (uint)group_size;

                for (uint i = 0; i < (uint)group_size; i++) {
                    sum += (float)x[x_base + i] * (float)codebook[cb_base + i];
                }
            }

            uint out_idx = b * (uint)out_dim + od;
            result[out_idx] = sum;
        """

        kernel = mx.fast.metal_kernel(
            name="codebook_matvec_2d",
            input_names=[
                "x",
                "codebook",
                "indices",
                "batch_size",
                "out_dim",
                "n_groups",
                "group_size",
                "in_dim",
            ],
            output_names=["result"],
            source=source,
            ensure_row_contiguous=True,
        )
        return kernel

    def codebook_moe_matvec(
        self,
        x: mx.array,
        codebook: mx.array,
        indices: mx.array,
        selected_expert_ids: Union[mx.array, List[int]],
        out_dim: int,
        n_groups: int,
        group_size: int,
    ) -> mx.array:
        """
        Fused codebook matvec for MoE with top-k expert selection.

        Args:
            x: Input array [batch, in_dim]
            codebook: Codebook array [n_codes, group_size] float16
            indices: Indices array [n_experts, out_dim, n_groups] uint16
            selected_expert_ids: Selected expert indices [k] uint16 or list
            out_dim: Output dimension per expert
            n_groups: Number of codebook groups
            group_size: Elements per codebook entry (typically 8)

        Returns:
            Output array [batch, k, out_dim]
        """
        if "codebook_moe_matvec" not in self._kernels:
            return self._mlx_moe_matmul(
                x, codebook, indices, selected_expert_ids, out_dim, n_groups, group_size
            )

        # Handle input shape - x should be [batch, in_dim]
        x_shape = x.shape
        batch_size = x_shape[0]
        in_dim = x_shape[1]

        # Convert selected_expert_ids to mx.array
        if isinstance(selected_expert_ids, (list, tuple)):
            selected_expert_ids = np.array(selected_expert_ids, dtype=np.uint16)
        if isinstance(selected_expert_ids, np.ndarray):
            selected_expert_ids = mx.array(selected_expert_ids, dtype=mx.uint16)
        elif (
            hasattr(selected_expert_ids, "dtype")
            and selected_expert_ids.dtype != mx.uint16
        ):
            selected_expert_ids = selected_expert_ids.astype(mx.uint16)

        k = selected_expert_ids.shape[0]

        out_shape = (batch_size, k, out_dim)

        try:
            out = self._kernels["codebook_moe_matvec"](
                inputs=[
                    x,  # [batch, in_dim]
                    codebook,  # [n_codes, group_size]
                    indices,  # [n_experts, out_dim, n_groups]
                    selected_expert_ids,  # [k]
                    mx.array(batch_size, dtype=mx.uint32),
                    mx.array(k, dtype=mx.uint32),
                    mx.array(out_dim, dtype=mx.uint32),
                    mx.array(n_groups, dtype=mx.uint32),
                    mx.array(group_size, dtype=mx.uint32),
                    mx.array(in_dim, dtype=mx.uint32),
                ],
                output_shapes=[out_shape],
                output_dtypes=[x.dtype],
                grid=(batch_size, k, out_dim),
                threadgroup=(1, 1, 1),
                verbose=False,
            )
            return out[0]
        except Exception as e:
            logger.warning(f"Kernel call failed: {e}, falling back to MLX")
            import traceback

            traceback.print_exc()
            return self._mlx_moe_matmul(
                x, codebook, indices, selected_expert_ids, out_dim, n_groups, group_size
            )

        # Determine input shape
        x_shape = x.shape
        if len(x_shape) == 2:
            batch_size = x_shape[0]
            in_dim = x_shape[1]
            k = 1
            x = x.reshape(batch_size, 1, in_dim)
        else:
            batch_size = x_shape[0]
            k = x_shape[1]
            in_dim = x_shape[2]

        # Convert selected_expert_ids to mx.array
        if isinstance(selected_expert_ids, (list, tuple)):
            selected_expert_ids = np.array(selected_expert_ids, dtype=np.uint16)
        if isinstance(selected_expert_ids, np.ndarray):
            selected_expert_ids = mx.array(selected_expert_ids, dtype=mx.uint16)
        elif (
            hasattr(selected_expert_ids, "dtype")
            and selected_expert_ids.dtype != mx.uint16
        ):
            selected_expert_ids = selected_expert_ids.astype(mx.uint16)

        out_shape = (batch_size, k, out_dim)

        try:
            out = self._kernels["codebook_moe_matvec"](
                inputs=[
                    x,
                    codebook,
                    indices,
                    selected_expert_ids,
                    mx.array(batch_size, dtype=mx.uint32),
                    mx.array(k, dtype=mx.uint32),
                    mx.array(out_dim, dtype=mx.uint32),
                    mx.array(n_groups, dtype=mx.uint32),
                    mx.array(group_size, dtype=mx.uint32),
                    mx.array(in_dim, dtype=mx.uint32),
                ],
                output_shapes=[out_shape],
                output_dtypes=[x.dtype],
                grid=(batch_size, k, out_dim),
                threadgroup=(1, 1, 1),
                verbose=False,
            )
            return out[0]
        except Exception as e:
            logger.warning(f"Kernel call failed: {e}, falling back to MLX")
            import traceback

            traceback.print_exc()
            return self._mlx_moe_matmul(
                x, codebook, indices, selected_expert_ids, out_dim, n_groups, group_size
            )

    def codebook_matvec_2d(
        self,
        x: mx.array,
        codebook: mx.array,
        indices: mx.array,
        out_dim: int,
        n_groups: int,
        group_size: int,
    ) -> mx.array:
        """
        Fused codebook matvec with 2D indices (simple codebook VQ).
        """
        if "codebook_matvec_2d" not in self._kernels:
            return self._mlx_matvec_2d(
                x, codebook, indices, out_dim, n_groups, group_size
            )

        batch_size = x.shape[0]
        in_dim = n_groups * group_size

        out_shape = (batch_size, out_dim)

        try:
            out = self._kernels["codebook_matvec_2d"](
                inputs=[
                    x,
                    codebook,
                    indices,
                    mx.array(batch_size, dtype=mx.uint32),
                    mx.array(out_dim, dtype=mx.uint32),
                    mx.array(n_groups, dtype=mx.uint32),
                    mx.array(group_size, dtype=mx.uint32),
                    mx.array(in_dim, dtype=mx.uint32),
                ],
                output_shapes=[out_shape],
                output_dtypes=[x.dtype],
                grid=(batch_size, out_dim, 1),
                threadgroup=(1, 1, 1),
                verbose=False,
            )
            return out[0]
        except Exception as e:
            logger.warning(f"Kernel call failed: {e}, falling back to MLX")
            return self._mlx_matvec_2d(
                x, codebook, indices, out_dim, n_groups, group_size
            )

    def _mlx_moe_matmul(
        self,
        x: mx.array,
        codebook: mx.array,
        indices: mx.array,
        selected_expert_ids,
        out_dim: int,
        n_groups: int,
        group_size: int,
    ) -> mx.array:
        """MLX fallback for MoE codebook matmul."""
        if isinstance(selected_expert_ids, (list, tuple)):
            selected_expert_ids = mx.array(selected_expert_ids, dtype=mx.uint32)
        elif not isinstance(selected_expert_ids, mx.array):
            selected_expert_ids = mx.array(selected_expert_ids, dtype=mx.uint32)

        batch_size = x.shape[0]
        k = selected_expert_ids.shape[0]

        outputs = []
        for i in range(k):
            expert_id = int(selected_expert_ids[i])
            expert_indices = indices[expert_id]

            flat_idx = expert_indices.reshape(-1)
            flat_weights = codebook[flat_idx]
            W = flat_weights.reshape(out_dim, -1)

            out = x @ W.T
            outputs.append(out)

        return mx.stack(outputs, axis=1)

    def _mlx_matvec_2d(
        self,
        x: mx.array,
        codebook: mx.array,
        indices: mx.array,
        out_dim: int,
        n_groups: int,
        group_size: int,
    ) -> mx.array:
        """MLX fallback for 2D codebook matmul."""
        flat_idx = indices.reshape(-1)
        flat_weights = codebook[flat_idx]
        W = flat_weights.reshape(out_dim, n_groups * group_size)

        if not W.flags.c_contiguous and not W.flags.f_contiguous:
            W = mx.contiguous(W)

        return x @ W.T

    @property
    def is_available(self) -> bool:
        """Check if Metal kernels are available and compiled."""
        return self._metal_available and len(self._kernels) > 0

    @property
    def device_name(self) -> Optional[str]:
        """Get the Metal device name."""
        try:
            d = mx.default_device()
            return str(d)
        except:
            return None

    def get_stats(self) -> dict:
        """Get kernel manager statistics."""
        return {
            "is_available": self.is_available,
            "metal_available": self._metal_available,
            "compiled_kernels": list(self._kernels.keys()),
            "threads_per_threadgroup": self._threads_per_threadgroup,
        }

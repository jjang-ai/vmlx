"""
Codebook Expert Loader - MoE integration for codebook VQ models.

Key design principle: SELECT FIRST, COMPUTE SECOND

Instead of:
  1. Load ALL expert indices [512, out_dim, n_groups]
  2. Reconstruct ALL expert weights [512, out_dim, in_dim]
  3. Select top-8 from reconstructed weights

We do:
  1. Get top-8 expert IDs from gate routing
  2. For each selected expert, get their indices slice [out_dim, n_groups]
  3. Fused kernel: lookup codebook entries + compute matmul in ONE pass
  4. Never materialize the full [512, out_dim, in_dim] weight tensor

This reduces memory from ~720GB (512 experts × 60 layers × 3 projections)
to ~1GB (8 experts × 60 layers × 3 projections) per forward pass.

Integration with mlx_lm MoE:
  - SwitchGLU contains gate routing + SwitchLinear
  - Gate selects top-8 of 512 experts
  - SwitchLinear does: for each selected expert, compute x @ W_e.T
  - We patch SwitchLinear to use codebook_moe_matvec instead of mx.gather_mm
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    mx = None


class CodebookExpertLoader:
    """
    Loads expert weights from codebook files for MoE layers.

    This class handles the integration between:
    1. The base model (which has placeholder expert weights)
    2. The codebook files (which have actual expert weights via indices)
    3. The forward pass (which needs real expert weights)

    Key optimization: Only loads indices for SELECTED experts (top-8 of 512),
    not all experts. Then uses fused kernel for codebook lookup + matmul.
    """

    def __init__(
        self,
        base_model: Any,
        model_path: Union[str, Path],
        jang_config: Dict[str, Any],
        lazy: bool = True,
        kernel_manager: Any = None,
    ):
        self._model = base_model
        self._path = Path(model_path)
        self._jang_config = jang_config
        self._lazy = lazy

        # Codebook config
        self._n_codes = jang_config.get("quantization", {}).get("n_codes", 16384)
        self._group_size = jang_config.get("quantization", {}).get(
            "codebook_group_size", 8
        )
        self._layer_metadata: Dict[str, Dict] = jang_config.get("codebook_layers", {})

        # Kernel manager for fused matmul
        self._kernel_manager = kernel_manager

        # Index codebook files: (layer_idx, tensor_type) -> Path
        self._codebook_files: Dict[Tuple[int, str], Path] = {}
        self._index_codebook_files()

        # Cache for loaded codebook data: (layer_idx, tensor_type) -> (codebook, indices)
        self._codebook_cache: Dict[Tuple[int, str], Tuple[mx.array, mx.array]] = {}

        # Cache for selected expert indices: (layer_idx, tensor_type, expert_id) -> indices_slice
        # This avoids re-extracting indices for the same expert
        self._expert_indices_cache: Dict[Tuple[int, str, int], mx.array] = {}

        # Stats
        self._codebook_loads = 0
        self._expert_selections = 0
        self._kernel_calls = 0

        logger.info(
            f"CodebookExpertLoader initialized: {len(self._codebook_files)} codebook files, "
            f"n_codes={self._n_codes}, group_size={self._group_size}"
        )

    def _index_codebook_files(self):
        """Index all codebook-layer-{NNN}-{type}.safetensors files."""
        for f in self._path.glob("codebook-layer-*.safetensors"):
            # Parse: codebook-layer-{NNN}-{type}.safetensors
            name = f.stem
            parts = name.split("-")
            if len(parts) >= 4:
                layer_idx = int(parts[2])
                tensor_type = parts[3]
                self._codebook_files[(layer_idx, tensor_type)] = f

        logger.info(f"Indexed {len(self._codebook_files)} codebook files")

    def load_codebook_layer(
        self, layer_idx: int, tensor_type: str
    ) -> Tuple[mx.array, mx.array]:
        """
        Load codebook and indices for a layer/projection type.

        Returns:
            Tuple of (codebook, indices) as mx.arrays.
            codebook: [n_codes, group_size]
            indices: [n_experts, out_dim, n_groups]
        """
        key = (layer_idx, tensor_type)

        if key in self._codebook_cache:
            return self._codebook_cache[key]

        file_path = self._codebook_files.get(key)
        if file_path is None:
            raise FileNotFoundError(
                f"Codebook file not found: layer={layer_idx}, type={tensor_type}"
            )

        # Load safetensors with MLX (faster than numpy)
        data = mx.load(str(file_path))
        codebook = data["codebook"]
        indices = data["indices"]

        # Cache
        self._codebook_cache[key] = (codebook, indices)
        self._codebook_loads += 1

        logger.debug(
            f"Loaded codebook layer {layer_idx}/{tensor_type}: "
            f"codebook={codebook.shape}, indices={indices.shape}"
        )

        return codebook, indices

    def get_expert_indices_slice(
        self,
        layer_idx: int,
        tensor_type: str,
        expert_id: int,
    ) -> mx.array:
        """
        Get indices slice for a single expert.

        This extracts indices[expert_id, :, :] which is [out_dim, n_groups].
        We cache these slices to avoid re-extracting for the same expert.

        Args:
            layer_idx: Layer index
            tensor_type: "gate_proj", "up_proj", or "down_proj"
            expert_id: Expert index (0-511)

        Returns:
            Indices slice [out_dim, n_groups]
        """
        cache_key = (layer_idx, tensor_type, expert_id)

        if cache_key in self._expert_indices_cache:
            return self._expert_indices_cache[cache_key]

        # Load the full layer indices (this is cached at layer level)
        _, indices = self.load_codebook_layer(layer_idx, tensor_type)

        # Extract this expert's indices
        expert_indices = indices[expert_id]

        # Cache
        self._expert_indices_cache[cache_key] = expert_indices

        return expert_indices

    def codebook_moe_matmul(
        self,
        x: mx.array,
        layer_idx: int,
        tensor_type: str,
        selected_expert_ids: Union[List[int], mx.array],
    ) -> mx.array:
        """
        Fused codebook matmul for MoE selected experts.

        This is the key optimization: instead of reconstructing all 512 experts'
        weights and then selecting top-8, we:

        1. Get the selected expert IDs (top-8 from gate routing)
        2. For each selected expert, use the fused kernel to:
           a. Look up codebook entries using that expert's indices
           b. Compute dot product with input x
        3. All in ONE GPU pass without materializing full weight matrix

        Args:
            x: Input array [batch, in_dim]
            layer_idx: Layer index
            tensor_type: "gate_proj", "up_proj", or "down_proj"
            selected_expert_ids: List of selected expert indices (e.g., top-8 of 512)

        Returns:
            Output array [batch, k, out_dim] where k = len(selected_expert_ids)
        """
        t0 = time.time()

        # Load codebook and indices for this layer
        codebook, indices = self.load_codebook_layer(layer_idx, tensor_type)

        # Infer dimensions from shapes
        n_experts, out_dim, n_groups = indices.shape
        in_dim = n_groups * self._group_size

        # Ensure selected_expert_ids is an array
        if not isinstance(selected_expert_ids, mx.array):
            selected_expert_ids = mx.array(selected_expert_ids, dtype=mx.uint16)
        else:
            selected_expert_ids = selected_expert_ids.astype(mx.uint16)

        k = len(selected_expert_ids)

        # Use kernel manager for fused operation
        if self._kernel_manager is not None and self._kernel_manager.is_available:
            output = self._kernel_manager.codebook_moe_matvec(
                x=x,
                codebook=codebook,
                indices=indices,
                selected_expert_ids=selected_expert_ids,
                out_dim=out_dim,
                n_groups=n_groups,
                group_size=self._group_size,
            )
        else:
            # Fallback: MLX reference implementation
            output = self._mlx_moe_matmul(
                x=x,
                codebook=codebook,
                indices=indices,
                selected_expert_ids=selected_expert_ids,
                out_dim=out_dim,
                n_groups=n_groups,
                group_size=self._group_size,
            )

        self._expert_selections += k
        self._kernel_calls += 1

        elapsed = time.time() - t0
        logger.debug(
            f"codebook_moe_matmul({layer_idx}/{tensor_type}): "
            f"k={k}, time={elapsed * 1000:.2f}ms"
        )

        return output

    def _mlx_moe_matmul(
        self,
        x: mx.array,
        codebook: mx.array,
        indices: mx.array,
        selected_expert_ids: mx.array,
        out_dim: int,
        n_groups: int,
        group_size: int,
    ) -> mx.array:
        """
        MLX fallback for MoE codebook matmul.

        Reference implementation that reconstructs only selected experts.
        """
        batch_size = x.shape[0]
        k = len(selected_expert_ids)
        in_dim = n_groups * group_size

        outputs = []

        for i in range(k):
            expert_id = int(selected_expert_ids[i])

            # Get this expert's indices: [out_dim, n_groups]
            expert_indices = indices[expert_id]

            # Reconstruct this expert's weights: [out_dim, in_dim]
            # Flatten indices and lookup codebook entries
            flat_idx = expert_indices.reshape(-1)
            flat_weights = codebook[flat_idx]
            W = flat_weights.reshape(out_dim, in_dim)

            # Compute x @ W^T: [batch, out_dim]
            out = x @ W.T
            outputs.append(out)

        # Stack: [batch, k, out_dim]
        return mx.stack(outputs, axis=1)

    def patch_model(self):
        """
        Patch the model's SwitchLinear to use codebook matmul.

        This monkey-patches the mlx_lm MoE implementation to:
        1. Intercept the expert routing (top-8 selection)
        2. Use codebook_moe_matmul instead of mx.gather_mm + matmul
        """
        logger.info("Patching model SwitchLinear for codebook VQ")

        # Find SwitchLinear layers in the model
        self._patched_layers = []
        self._original_forward = {}

        for name, module in self._model.named_modules():
            if "switch_mlp" in name.lower() or "switch_linear" in name.lower():
                if hasattr(module, "forward") and not getattr(
                    module, "_codebook_patched", False
                ):
                    self._patch_switch_linear(name, module)
                    self._patched_layers.append(name)

        logger.info(
            f"Patched {len(self._patched_layers)} MoE layers: {self._patched_layers}"
        )

    def _patch_switch_linear(self, name: str, module: Any):
        """Patch a single SwitchLinear module."""
        original_forward = module.forward

        def codebook_forward(hidden_states: mx.array, expert_indices: mx.array = None):
            """
            Patched forward that uses codebook matmul.

            Args:
                hidden_states: [batch, in_dim]
                expert_indices: [k] selected expert indices (optional, will use routing if None)

            Returns:
                MoE output [batch, out_dim]
            """
            # Get layer info from module name
            layer_idx = self._extract_layer_idx(name)
            tensor_types = ["gate_proj", "up_proj", "down_proj"]

            # If expert_indices not provided, use routing
            if expert_indices is None:
                # Use original routing
                return original_forward(hidden_states)

            # Use codebook matmul for each projection
            outputs = []
            for tensor_type in tensor_types:
                out = self.codebook_moe_matmul(
                    x=hidden_states,
                    layer_idx=layer_idx,
                    tensor_type=tensor_type,
                    selected_expert_ids=expert_indices,
                )
                outputs.append(out)

            # outputs: [batch, k, out_dim] for each projection
            # Apply activation (SiLU for gate, GELU for up/down - handled by SwitchGLU)
            gate_out = outputs[0]  # gate_proj
            up_out = outputs[1]  # up_proj
            down_out = outputs[2]  # down_proj

            # SwitchGLU: gate * up @ down
            # gate: [batch, k, inter_dim], up: [batch, k, inter_dim], down: [batch, k, out_dim]
            # Result: [batch, k, out_dim]

            # Apply SiLU to gate
            gate_act = mx.silu(gate_out)

            # Element-wise multiply gate * up
            gate_up = gate_act * up_out

            # Matmul with down: [batch, k, inter_dim] @ [batch, k, out_dim]
            # This is a batched matmul
            down_out_T = down_out.transpose(0, 2, 1)  # [batch, out_dim, k]
            result = gate_up @ down_out_T  # [batch, k, out_dim]

            # Sum over experts (weighted by routing scores - handled by caller)
            return result.sum(axis=1)  # [batch, out_dim]

        module.forward = codebook_forward
        module._codebook_patched = True
        self._original_forward[name] = original_forward

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from module name."""
        import re

        match = re.search(r"layers\.(\d+)", name)
        if match:
            return int(match.group(1))
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            "codebook_loads": self._codebook_loads,
            "expert_selections": self._expert_selections,
            "kernel_calls": self._kernel_calls,
            "cached_codebooks": len(self._codebook_cache),
            "cached_expert_indices": len(self._expert_indices_cache),
            "codebook_files": len(self._codebook_files),
            "kernel_available": (
                self._kernel_manager.is_available if self._kernel_manager else False
            ),
        }

    def clear_cache(self):
        """Clear all caches."""
        self._codebook_cache.clear()
        self._expert_indices_cache.clear()


def create_expert_loader(
    base_model: Any,
    model_path: Union[str, Path],
    jang_config: Dict[str, Any],
    lazy: bool = True,
    kernel_manager: Any = None,
) -> CodebookExpertLoader:
    """Create and return a CodebookExpertLoader."""
    return CodebookExpertLoader(
        base_model=base_model,
        model_path=model_path,
        jang_config=jang_config,
        lazy=lazy,
        kernel_manager=kernel_manager,
    )

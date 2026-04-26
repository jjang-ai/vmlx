"""
Codebook MoE Integration - Patch SwitchLinear to use codebook VQ.

This module patches mlx_lm's SwitchLinear/QuantizedSwitchLinear at the CLASS level
to use CodebookExpertLoader for expert weight loading when codebook VQ is enabled.

Key insight: Python calls type(obj).__call__(obj, *args) not obj.__call__(*args),
so we must patch at the class level.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    mx = None


# Global reference to expert loader (set by patch_model)
_expert_loader: Optional[Any] = None
_original_calls: Dict[type, callable] = {}


def _patched_switch_linear_call(self, x, indices, sorted_indices=False):
    """
    Patched __call__ for SwitchLinear/QuantizedSwitchLinear.

    Checks if this instance has codebook info. If so, uses codebook loader.
    Otherwise, falls back to original implementation.
    """
    # Check if this instance has codebook info
    if not hasattr(self, "_codebook_layer_idx") or not hasattr(
        self, "_codebook_tensor_type"
    ):
        # Not a codebook-patched instance, use original
        return _original_calls[type(self)](self, x, indices, sorted_indices)

    layer_idx = self._codebook_layer_idx
    tensor_type = self._codebook_tensor_type
    loader = self._codebook_loader

    # Check if codebook file exists
    if not loader._codebook_files.get((layer_idx, tensor_type)):
        return _original_calls[type(self)](self, x, indices, sorted_indices)

    # Handle indices shape FIRST
    # indices can be [batch, seq_len, k] or [k] depending on routing
    indices_ndim = len(indices.shape)
    if indices_ndim == 1:
        unique_experts = indices.tolist()
        k = len(unique_experts)
        batch_from_indices = 1
        seq_len_from_indices = 1
    elif indices_ndim == 2:
        unique_experts = indices[0].tolist()
        k = len(unique_experts)
        batch_from_indices = indices.shape[0]
        seq_len_from_indices = indices.shape[1]
    else:
        unique_experts = indices[0, 0].tolist()
        k = len(unique_experts)
        batch_from_indices = indices.shape[0]
        seq_len_from_indices = indices.shape[1]

    # Handle input shapes
    # x shape depends on which projection:
    # - gate/up: x is 5D (batch, seq_len, 1, 1, input_dim) with k=1
    # - down_proj: x is 5D (batch, seq_len, k, 1, input_dim) with k=experts
    original_shape = x.shape
    ndim = len(x.shape)
    if ndim == 5:
        # x is 5D from SwitchGLU: (batch, seq_len, k_or_1, 1, input_dim)
        # The 3rd dimension is k (experts) - it's 1 for gate/up, >1 for down
        batch = x.shape[0]
        seq_len = x.shape[1]
        k_or_1 = x.shape[2]
        input_dim = x.shape[4]
        if k_or_1 == 1:
            # gate/up: x = (batch, seq_len, 1, 1, input_dim)
            # Flatten to (batch * seq_len, input_dim)
            x_flat = x.reshape(batch * seq_len, input_dim)
        else:
            # down_proj: x = (batch, seq_len, k, 1, input_dim)
            # Each of the k expert outputs needs to be processed by k experts
            # x_flat should be (batch * seq_len, input_dim) - NOT (batch * seq_len * k, input_dim)
            # because each expert output is processed separately
            x_flat = x[:, :, 0, 0, :].reshape(batch * seq_len, input_dim)
    elif ndim == 3:
        # [batch, seq_len, input_dim] - standard case
        batch, seq_len, input_dim = x.shape
        x_flat = x.reshape(-1, input_dim)
    elif ndim == 4:
        # [batch, seq_len, k, input_dim] - 4D case (unlikely but handle it)
        batch, seq_len, k, input_dim = x.shape
        x_flat = x.reshape(batch * seq_len * k, input_dim)
    elif ndim == 2:
        batch = 1
        seq_len = x.shape[0]
        input_dim = x.shape[1]
        x_flat = x
    else:
        # Handle 6D or higher
        batch = batch_from_indices
        seq_len = seq_len_from_indices
        input_dim = x.shape[-1]
        x_flat = x.reshape(-1, input_dim)

    # Compute using codebook loader
    output = loader.codebook_moe_matmul(
        x=x_flat,
        layer_idx=layer_idx,
        tensor_type=tensor_type,
        selected_expert_ids=unique_experts,
    )

    # Reshape output to [batch, seq_len, k, 1, output_dim]
    # SwitchLinear expects 5D output with an extra dimension of size 1
    output_dim = output.shape[-1]
    actual_first_dim = output.shape[0]
    expected_first_dim = batch * seq_len
    if actual_first_dim == expected_first_dim:
        output = output.reshape(batch, seq_len, k, 1, output_dim)
        logger.debug(f"Reshaped to: {output.shape}")
    else:
        logger.warning(
            f"Unexpected output shape {output.shape}, expected [{expected_first_dim}, {k}, {output_dim}]"
        )
        if actual_first_dim == k and expected_first_dim == 1:
            output = output.reshape(1, seq_len, k, 1, output_dim)
        else:
            output = output.reshape(1, seq_len, k, 1, output_dim)
        logger.debug(f"Emergency reshape to: {output.shape}")

    return output


class CodebookMoEIntegration:
    """
    Integrates codebook VQ with mlx_lm MoE models.

    Patches SwitchLinear/QuantizedSwitchLinear at the class level
    to use CodebookExpertLoader for expert weight loading.
    """

    def __init__(
        self,
        expert_loader: Any,
        verbose: bool = False,
    ):
        self._loader = expert_loader
        self._verbose = verbose
        self._patched_classes: List[type] = []
        self._instances_patched: int = 0

    def _find_switch_linear_classes(self) -> List[type]:
        """Find SwitchLinear and QuantizedSwitchLinear classes."""
        from mlx_lm.models import switch_layers

        return [
            getattr(switch_layers, name)
            for name in ["SwitchLinear", "QuantizedSwitchLinear"]
            if hasattr(switch_layers, name)
        ]

    def patch_model(self, model: Any) -> int:
        """
        Patch all SwitchLinear/QuantizedSwitchLinear classes.

        Args:
            model: The mlx_lm model

        Returns:
            Number of classes patched
        """
        global _expert_loader, _original_calls

        _expert_loader = self._loader

        # Find classes to patch
        classes = self._find_switch_linear_classes()
        logger.info(f"Found {len(classes)} SwitchLinear classes to patch")

        # Store originals and patch
        for cls in classes:
            if cls in _original_calls:
                continue  # Already patched

            _original_calls[cls] = cls.__call__
            cls.__call__ = _patched_switch_linear_call
            self._patched_classes.append(cls)
            logger.info(f"Patched {cls.__name__}.__call__")

        # Now mark specific instances
        self._instances_patched = self._mark_instances(model)
        logger.info(f"Marked {self._instances_patched} instances for codebook loading")

        return len(self._patched_classes)

    def _mark_instances(self, model: Any) -> int:
        """Mark specific instances with their layer/tensor info."""
        count = 0

        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            if class_name not in ("SwitchLinear", "QuantizedSwitchLinear"):
                continue

            # Extract layer index and tensor type
            layer_idx = None
            tensor_type = None

            layer_match = re.search(r"layers\.(\d+)", name)
            if layer_match:
                layer_idx = int(layer_match.group(1))

            for tt in ["gate_proj", "up_proj", "down_proj"]:
                if f".{tt}" in name:
                    tensor_type = tt
                    break

            if layer_idx is not None and tensor_type is not None:
                # Mark this instance
                module._codebook_layer_idx = layer_idx
                module._codebook_tensor_type = tensor_type
                module._codebook_loader = _expert_loader
                count += 1

                if self._verbose:
                    logger.info(f"Marked {name}: layer={layer_idx}, type={tensor_type}")

        return count

    def unpatch_all(self):
        """Restore original __call__ methods."""
        global _original_calls

        for cls in self._patched_classes:
            if cls in _original_calls:
                cls.__call__ = _original_calls[cls]
                logger.info(f"Unpatched {cls.__name__}")

        self._patched_classes = []
        _original_calls = {}


def patch_switch_linear(
    model: Any,
    expert_loader: Any,
    verbose: bool = False,
) -> CodebookMoEIntegration:
    """
    Patch SwitchLinear layers in model to use codebook loader.

    Args:
        model: The mlx_lm model
        expert_loader: CodebookExpertLoader instance
        verbose: Enable verbose logging

    Returns:
        CodebookMoEIntegration instance
    """
    integration = CodebookMoEIntegration(expert_loader, verbose)
    integration.patch_model(model)
    return integration


def find_switch_linears(model: Any) -> List[Dict]:
    """Find all SwitchLinear modules in a model."""
    results = []

    for name, module in model.named_modules():
        if not hasattr(module, "__class__"):
            continue

        class_name = module.__class__.__name__
        if class_name not in ("SwitchLinear", "QuantizedSwitchLinear"):
            continue

        layer_idx = None
        tensor_type = None

        layer_match = re.search(r"layers\.(\d+)", name)
        if layer_match:
            layer_idx = int(layer_match.group(1))

        for tt in ["gate_proj", "up_proj", "down_proj"]:
            if f".{tt}" in name:
                tensor_type = tt
                break

        results.append(
            {
                "name": name,
                "layer_idx": layer_idx,
                "tensor_type": tensor_type,
                "module": module,
            }
        )

    return results

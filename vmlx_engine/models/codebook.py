"""
Codebook VQ Language Model - MLX model wrapper with codebook expert weights.

This class wraps an MLX model with codebook VQ expert weights, providing:
- Lazy loading of codebook files
- Automatic weight reconstruction on first access
- Fused Metal kernel for codebook matmul
- Integration with vMLX scheduler and cache

Usage:
    model = CodebookVQLanguageModel(
        model_path="/path/to/Qwen3.5-35B-A3B-CODEBOOK-TEST",
        config=config_manager
    )

    # Forward pass uses codebook matmul automatically
    output = model(inputs)
"""

import json
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


class CodebookVQLanguageModel:
    """
    MLX language model wrapper with codebook VQ expert weights.

    Supports lazy loading of codebook files and automatic weight reconstruction
    using fused Metal kernels for performance.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        base_model: Any,
        tokenizer: Any,
        jang_config: Dict[str, Any],
        config_manager: Any,
    ):
        """
        Initialize codebook VQ model wrapper.

        Args:
            model_path: Path to the model directory.
            base_model: Pre-loaded MLX model (non-expert weights).
            tokenizer: Tokenizer instance.
            jang_config: JANG config dict with codebook_vq settings.
            config_manager: ConfigManager instance for settings.
        """
        self.model_path = Path(model_path)
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._jang_config = jang_config
        self._config = config_manager

        # Codebook configuration
        self._n_codes = jang_config.get("quantization", {}).get("n_codes", 16384)
        self._group_size = jang_config.get("quantization", {}).get(
            "codebook_group_size", 8
        )

        # Codebook file index: (layer_idx, tensor_type) -> file_path
        self._codebook_files: Dict[Tuple[int, str], Path] = {}
        self._codebook_metadata: Dict[str, Any] = {}

        # Reconstructed weights cache
        from vmlx_engine.cache.codebook_cache import CodebookWeightCache

        # Handle None config_manager
        if config_manager is not None:
            cache_config = config_manager.get("memory.codebook_cache", {})
            use_metal = config_manager.get("codebook.kernel", "metal") == "metal"
            kernel_config = config_manager.get("codebook.kernel_config", {})
        else:
            cache_config = {}
            use_metal = True  # Default to metal if no config
            kernel_config = {}

        self._weight_cache = CodebookWeightCache(
            config=cache_config,
            disk_cache_dir=cache_config.get("disk_cache_dir"),
            disk_max_gb=cache_config.get("disk_max_gb", 1000),
            eviction_batch_size=cache_config.get("eviction_batch_size", 4),
        )

        # Kernel manager for fused matmul
        if use_metal and _MLX_AVAILABLE:
            from vmlx_engine.metal.kernel_manager import CodebookKernelManager

            self._kernel_manager = CodebookKernelManager(
                threads_per_threadgroup=kernel_config.get(
                    "threads_per_threadgroup", 256
                ),
                max_batch_size=kernel_config.get("max_batch_size", 32),
            )
        else:
            self._kernel_manager = None

        # Index codebook files
        self._index_codebook_files()

        # Load codebook metadata from jang_config
        self._codebook_metadata = jang_config.get("codebook_layers", {})

        # Stats
        self._forward_count = 0
        self._total_matmul_time = 0.0

        logger.info(
            f"CodebookVQLanguageModel initialized: {len(self._codebook_files)} codebook files, "
            f"n_codes={self._n_codes}, group_size={self._group_size}"
        )

    def _index_codebook_files(self):
        """Index all codebook-layer-{NNN}-{type}.safetensors files."""
        for f in self.model_path.glob("codebook-layer-*.safetensors"):
            # Parse filename: codebook-layer-{NNN}-{type}.safetensors
            name = f.stem  # e.g., "codebook-layer-000-gate_proj"
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
        Load and reconstruct one layer's expert weights.

        Args:
            layer_idx: Layer index.
            tensor_type: Tensor type ("gate_proj", "up_proj", "down_proj").

        Returns:
            Tuple of (codebook, indices) as mx.arrays.
        """
        key = (layer_idx, tensor_type)

        # Check cache first
        cached_weights = self._weight_cache.get(key)
        if cached_weights is not None:
            return cached_weights

        # Load from file
        file_path = self._codebook_files.get(key)
        if file_path is None:
            raise FileNotFoundError(f"Codebook file not found for {key}")

        # Load safetensors
        data = mx.load(str(file_path))
        codebook = data["codebook"]
        indices = data["indices"]

        # Store reconstructed weights in cache
        self._weight_cache.put(key, (codebook, indices))

        return codebook, indices

    def codebook_matmul(
        self,
        x: mx.array,
        layer_idx: int,
        tensor_type: str,
    ) -> mx.array:
        """
        Compute matmul with codebook VQ expert weights.

        This uses either:
        - Fused Metal kernel (if available and enabled)
        - Reference MLX implementation

        Args:
            x: Input array [batch, in_dim].
            layer_idx: Layer index.
            tensor_type: Tensor type ("gate_proj", "up_proj", "down_proj").

        Returns:
            Output array [batch, out_dim].
        """
        start_time = time.time()

        # Load codebook layer (lazy, cached)
        codebook, indices = self.load_codebook_layer(layer_idx, tensor_type)

        if self._kernel_manager is not None:
            # Use fused Metal kernel
            output = self._kernel_manager.codebook_matvec(
                x, codebook, indices, self._group_size
            )
        else:
            # Reference MLX implementation
            output = self._reference_matmul(x, codebook, indices, self._group_size)

        elapsed = time.time() - start_time
        self._total_matmul_time += elapsed
        self._forward_count += 1

        return output

    def _reference_matmul(
        self,
        x: mx.array,
        codebook: mx.array,
        indices: mx.array,
        group_size: int,
    ) -> mx.array:
        """
        Reference MLX implementation of codebook matmul.

        This reconstructs the weights from codebook + indices, then does standard matmul.
        Used as fallback when Metal kernel is not available.

        Args:
            x: Input array [batch, in_dim].
            codebook: Codebook array [n_codes, group_size].
            indices: Indices array [out_dim, n_groups].
            group_size: Elements per codebook entry.

        Returns:
            Output array [batch, out_dim].
        """
        # Flatten indices and lookup codebook entries
        flat_idx = indices.reshape(-1)
        flat_weights = codebook[flat_idx]

        # Reshape to weight matrix
        out_dim, n_groups = indices.shape
        in_dim = n_groups * group_size
        W = flat_weights.reshape(out_dim, in_dim)

        # Matmul: x @ W^T
        return x @ W.T

    def __call__(self, *args, **kwargs) -> mx.array:
        """
        Forward pass - delegates to base model.

        The base model's forward pass will call codebook_matmul for expert layers.
        """
        return self._base_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to base model."""
        return getattr(self._base_model, name)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_matmul_time = (
            self._total_matmul_time / self._forward_count
            if self._forward_count > 0
            else 0
        )

        return {
            "forward_count": self._forward_count,
            "total_matmul_time": self._total_matmul_time,
            "avg_matmul_time": avg_matmul_time,
            "weight_cache_stats": self._weight_cache.get_stats(),
        }

    def clear_cache(self):
        """Clear the weight cache."""
        self._weight_cache.clear()

    @property
    def weight_cache(self):
        """Return the weight cache for direct access."""
        return self._weight_cache

    @property
    def base_model(self):
        """Return the underlying base model."""
        return self._base_model

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

# SPDX-License-Identifier: Apache-2.0
"""
MLX Model wrappers for vLLM.

This module provides wrappers around mlx-lm and mlx-vlm for
integration with vLLM's model execution system.
"""

from vllm_mlx.models.llm import MLXLanguageModel
from vllm_mlx.models.mllm import MLXMultimodalLM

# Backwards compatibility aliases
MLXVisionLanguageModel = MLXMultimodalLM

__all__ = ["MLXLanguageModel", "MLXMultimodalLM", "MLXVisionLanguageModel"]

# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 JANGTQ loader — thin re-export.

Mirrors the pattern used for Kimi K2.6 (``load_jangtq_kimi_vlm``): a one-line
indirection so the research doc's code snippets work verbatim against the
vMLX engine (``from vmlx_engine.loaders.load_jangtq_dsv4 import
load_jangtq_dsv4_model``) AND so the bundled Python distribution always
exposes a stable import path even if ``jang_tools.dsv4`` internals move.

Under the hood this delegates to ``jang_tools.load_jangtq.load_jangtq_model``,
which already detects ``model_type="deepseek_v4"`` from ``config.json`` and
routes through ``jang_tools.dsv4.mlx_register`` (auto-imported at first call
via ``vmlx_engine.utils.jang_loader``) to register our custom MLX model class
(``jang_tools.dsv4.mlx_model.Model``) into ``mlx_lm.models`` namespace.

Runtime contract implemented by the underlying loader:
- mHC (Manifold Hyper-Connections) hc_mult=4 with 20 Sinkhorn iterations
  (fused Metal kernel, fallback pure-MLX op available).
- MLA attention head_dim=512 with grouped O projection (o_groups=8,
  o_lora_rank=1024), per-head RMSNorm, attention sink, inverse RoPE on
  output, partial RoPE over last qk_rope_head_dim=64 dimensions.
- 256 routed experts top-6 + 1 shared (switch_mlp stacked across layers).
- sqrtsoftplus routing with hash-table (tid2eid) for the first 3 layers,
  biased top-6 argpartition + weight-norm + routed_scaling_factor=1.5 for
  the remaining 40 layers.
- ``DeepseekV4Cache`` (RotatingKVCache sliding_window=128 + compressor and
  indexer state buffers). ``compress_ratios`` array per layer switches
  between plain attention and compressed global context attention.
- ``swiglu_limit=10`` SwiGLU activation, fp32 lm_head matmul (4096
  contraction in bf16 drifts logits enough to flip arithmetic answers —
  see research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md).

The function returns ``(model, tokenizer)`` just like the sibling loaders.
"""

from __future__ import annotations

from typing import Any, Tuple


def load_jangtq_dsv4_model(model_path: str, *, skip_params_eval: bool = False) -> Tuple[Any, Any]:
    """Load a DeepSeek V4 JANGTQ bundle.

    The actual work happens in ``jang_tools.load_jangtq.load_jangtq_model``.
    This wrapper exists for API-stability so the research docs'
    ``from vmlx_engine.loaders.load_jangtq_dsv4 import ...`` examples work.

    Args:
        model_path: Path to the bundle directory containing ``config.json``,
            ``jang_config.json``, and packed safetensors shards.
        skip_params_eval: If True, skip the post-load params materialization
            step in the underlying JANGTQ loader (for callers planning their
            own layer-by-layer warmup).

    Returns:
        A ``(model, tokenizer)`` tuple. ``model.config`` carries the raw
        ``config.json`` dict; ``tokenizer.jang_chat`` carries the
        ``jang_config.json.chat`` block (EOS + reasoning modes +
        tool_calling parser name + sampling_defaults).

    Raises:
        ImportError: if ``jang_tools.dsv4`` is not installed (older jang-tools
            without the dsv4 submodule). Callers should surface this as
            "Reinstall vMLX from the latest DMG — the bundled Python is out
            of date".
    """
    # Eagerly register mlx_lm.models.deepseek_v4 so the underlying loader
    # can resolve the model_type. Safe to call multiple times (idempotent).
    from jang_tools.dsv4 import mlx_register  # noqa: F401
    from jang_tools.load_jangtq import load_jangtq_model

    return load_jangtq_model(model_path, skip_params_eval=skip_params_eval)

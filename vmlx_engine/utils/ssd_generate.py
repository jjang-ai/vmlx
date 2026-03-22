# SPDX-License-Identifier: Apache-2.0
"""
Custom text generation loop for SSD disk-streaming inference.

Replaces mlx_lm.stream_generate() when SSD disk streaming is active.
During decode, decomposes the model forward pass into per-layer steps:
  embed -> (load layer i -> compute -> free) x N -> norm -> lm_head -> sample

This keeps only ONE transformer layer's weights in Metal GPU memory at a time,
enabling models larger than physical RAM to generate text from SSD at ~2-5 tok/s.

Prefill (prompt processing) uses the standard model forward pass with all weights
loaded, then frees all layer weights before entering the decode loop.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.utils.streaming_wrapper import _find_layers
from vmlx_engine.utils.weight_index import (
    build_weight_index,
    free_layer_weights,
    load_layer_weights,
)

logger = logging.getLogger(__name__)

# mx.eval is mlx.core.eval — a GPU synchronization barrier that forces Metal
# to execute the pending compute graph. This is NOT Python's builtins.eval().
_gpu_sync = mx.eval  # noqa: S307


def _find_model_components(model) -> dict[str, Any]:
    """Find the inner model components for decomposed forward pass.

    Returns dict with keys:
        embed_tokens, layers, norm, lm_head, tie_word_embeddings

    Raises:
        RuntimeError: If essential components cannot be found.
    """
    components: dict[str, Any] = {
        "embed_tokens": None,
        "layers": None,
        "norm": None,
        "lm_head": None,
        "tie_word_embeddings": False,
    }

    # Find layers via _find_layers (handles all model variants)
    result = _find_layers(model)
    if result is None:
        raise RuntimeError("Cannot find transformer layers on model")
    container, attr_name = result
    components["layers"] = getattr(container, attr_name)

    # The container holding layers also holds embed_tokens and norm.
    # Try paths in order of specificity:
    #   VLM:      model.language_model.model
    #   Standard: model.model
    #   Nemotron: model.backbone
    #   Direct:   model
    # Find the inner module that contains embed_tokens/embeddings and norm.
    # Different architectures use different attribute names.
    _embed_names = ("embed_tokens", "embeddings", "embedding", "wte")
    _norm_names = ("norm", "norm_f", "final_layernorm", "final_norm", "ln_f")

    inner = None
    embed_attr = None
    for path in [
        lambda m: getattr(getattr(m, "language_model", None), "model", None),
        lambda m: getattr(m, "model", None),
        lambda m: getattr(m, "backbone", None),
        lambda m: m,
    ]:
        candidate = path(model)
        if candidate is None:
            continue
        for ename in _embed_names:
            if hasattr(candidate, ename):
                inner = candidate
                embed_attr = ename
                break
        if inner is not None:
            break

    if inner is None or embed_attr is None:
        raise RuntimeError(
            "Cannot find embedding layer on model -- unsupported architecture"
        )

    components["embed_tokens"] = getattr(inner, embed_attr)

    # norm -- try multiple names
    norm_found = False
    for nname in _norm_names:
        if hasattr(inner, nname):
            components["norm"] = getattr(inner, nname)
            norm_found = True
            break
    if not norm_found:
        raise RuntimeError("Cannot find norm layer on model")

    # lm_head -- check for tied embeddings
    tie = getattr(getattr(model, "args", None), "tie_word_embeddings", False)
    if hasattr(model, "lm_head"):
        components["lm_head"] = model.lm_head
    elif tie and hasattr(components["embed_tokens"], "as_linear"):
        components["lm_head"] = components["embed_tokens"].as_linear
        components["tie_word_embeddings"] = True
    else:
        # VLM: model.language_model.lm_head
        lm = getattr(model, "language_model", None)
        if lm is not None and hasattr(lm, "lm_head"):
            components["lm_head"] = lm.lm_head
        elif hasattr(components["embed_tokens"], "as_linear"):
            components["lm_head"] = components["embed_tokens"].as_linear
            components["tie_word_embeddings"] = True
        else:
            raise RuntimeError("Cannot find lm_head on model")

    logger.debug(
        "Model components: %d layers, tie_embeddings=%s",
        len(components["layers"]),
        components["tie_word_embeddings"],
    )
    return components


def _create_attention_mask(
    h: mx.array, cache: list, offset: int = 0
) -> Optional[mx.array]:
    """Create the causal attention mask for a single decode step.

    Uses mlx_lm's create_attention_mask if available.

    Args:
        h: Hidden state tensor [batch, seq_len, hidden].
        cache: The KV cache list.
        offset: Token position offset (for manual mask creation).

    Returns:
        Attention mask array, or None if not needed (single token decode).
    """
    try:
        from mlx_lm.models.base import create_attention_mask

        return create_attention_mask(h, cache[0])
    except (ImportError, AttributeError, TypeError):
        # TypeError: custom cache classes (e.g. Nemotron's ArraysCache)
        # may not accept the same kwargs as standard KVCache.make_mask()
        pass

    # Fallback: for single-token decode, mask is typically not needed
    # by most architectures. If seq_len > 1, create a basic causal mask.
    seq_len = h.shape[1] if h.ndim >= 2 else 1
    if seq_len <= 1:
        return None

    # Basic causal mask
    mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
    return mask


def _load_layer_from_index(
    layer,
    layer_idx: int,
    model_path: Path,
    weight_index: dict,
    temp_weight_dir: Optional[Path],
) -> None:
    """Load a single layer's weights from disk.

    Args:
        layer: The layer module to load into.
        layer_idx: Index of the layer.
        model_path: Path to the original model files.
        weight_index: The weight index from build_weight_index().
        temp_weight_dir: If set, load from JANG temp files instead.

    Raises:
        RuntimeError: If weight loading fails.
    """
    if temp_weight_dir is not None:
        # JANG model -- load from temp files (relative keys, no prefix)
        # Try .npz first (handles all MLX dtypes), fall back to .safetensors
        temp_file = temp_weight_dir / f"layer_{layer_idx:04d}.npz"
        if not temp_file.exists():
            temp_file = temp_weight_dir / f"layer_{layer_idx:04d}.safetensors"
        if not temp_file.exists():
            raise RuntimeError(
                f"JANG temp weight file not found: {temp_file}"
            )
        load_layer_weights(layer, temp_file, weight_index_entry=None)
    else:
        # Standard model -- load from original safetensors via weight index
        if layer_idx not in weight_index:
            raise RuntimeError(
                f"Layer {layer_idx} not in weight index -- "
                f"available: {sorted(weight_index.keys())}"
            )
        entry = weight_index[layer_idx]
        for filename in entry["files"]:
            file_path = model_path / filename
            if not file_path.exists():
                raise RuntimeError(
                    f"Weight file not found: {file_path}"
                )
            load_layer_weights(layer, file_path, weight_index_entry=entry)


def ssd_stream_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, mx.array, list[int]],
    model_path: Union[str, Path],
    *,
    weight_index: Optional[dict] = None,
    temp_weight_dir: Optional[Union[str, Path]] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    prefetch_layers: int = 0,  # Reserved for future optimization
    **kwargs,
) -> Generator:
    """Stream-generate text with per-layer weight recycling from SSD.

    This is the core SSD disk-streaming generation loop. During decode,
    only one transformer layer's weights are in GPU memory at a time.

    Args:
        model: The loaded MLX model (with weights still in memory for prefill).
        tokenizer: The tokenizer (HF or TokenizerWrapper).
        prompt: Input text, token IDs, or mx.array.
        model_path: Path to model directory with safetensors files.
        weight_index: Pre-built weight index (if None, built automatically).
        temp_weight_dir: Directory with JANG temp layer files (if applicable).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling threshold.
        repetition_penalty: Repetition penalty factor (1.0 = disabled).
        prefetch_layers: Reserved for future async prefetching.

    Yields:
        GenerationResponse objects compatible with mlx_lm.stream_generate().
    """
    from mlx_lm.generate import GenerationResponse, make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    model_path = Path(model_path)
    if temp_weight_dir is not None:
        temp_weight_dir = Path(temp_weight_dir)

    # --- Wrap tokenizer ---
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # --- Tokenize prompt ---
    if isinstance(prompt, str):
        add_special_tokens = (
            tokenizer.bos_token is None
            or not prompt.startswith(tokenizer.bos_token)
        )
        prompt_tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        )
    elif isinstance(prompt, list):
        prompt_tokens = mx.array(prompt)
    else:
        prompt_tokens = prompt

    num_prompt_tokens = prompt_tokens.size

    # --- Build weight index ---
    if weight_index is None:
        weight_index = build_weight_index(model_path)

    # --- Create sampler ---
    sampler = make_sampler(temp=temperature, top_p=top_p)

    # --- Create KV cache ---
    prompt_cache = make_prompt_cache(model)

    # --- Find model components ---
    components = _find_model_components(model)
    embed_tokens = components["embed_tokens"]
    layers = components["layers"]
    norm = components["norm"]
    lm_head = components["lm_head"]
    num_layers = len(layers)

    # Check layer-to-cache mapping — hybrid SSM models may differ
    if len(prompt_cache) != num_layers:
        logger.warning(
            "SSD streaming: %d layers vs %d cache entries (hybrid SSM model). "
            "Falling back to standard generation with lazy-loaded (mmap) weights. "
            "macOS will page weights in/out from SSD automatically.",
            num_layers, len(prompt_cache),
        )
        # Fall back to mlx_lm.stream_generate — model's own __call__ handles
        # hybrid cache routing internally. Weights are still mmap'd from disk.
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.generate import make_sampler

        # mlx_lm.stream_generate passes **kwargs to generate_step which expects
        # a sampler callable, NOT temp/top_p params directly
        fallback_sampler = make_sampler(temp=temperature, top_p=top_p)

        for response in mlx_stream_generate(
            model, tokenizer,
            prompt if isinstance(prompt, str) else tokenizer.decode(prompt.tolist() if hasattr(prompt, 'tolist') else list(prompt)),
            max_tokens=max_tokens,
            sampler=fallback_sampler,
        ):
            yield response
        return

    # ===================================================================
    # HELPER: per-layer forward pass (used by both prefill and decode)
    # ===================================================================
    def _per_layer_forward(h: mx.array, mask):
        """Run h through all layers with per-layer weight recycling."""
        for i, (layer, cache_entry) in enumerate(zip(layers, prompt_cache)):
            _load_layer_from_index(
                layer, i, model_path, weight_index, temp_weight_dir
            )
            h = layer(h, mask, cache=cache_entry)
            _gpu_sync(h)
            free_layer_weights(layer)
        return h

    # ===================================================================
    # PREFILL — per-layer weight recycling (1 layer in memory at a time)
    # ===================================================================
    logger.info(
        "SSD prefill: %d prompt tokens, %d layers (per-layer)", num_prompt_tokens, num_layers
    )
    prefill_start = time.perf_counter()

    # Embed all prompt tokens
    h = embed_tokens(prompt_tokens[None])
    _gpu_sync(h)

    # Create mask from embedded prompt
    mask = _create_attention_mask(h, prompt_cache)

    # Process through all layers one at a time
    h = _per_layer_forward(h, mask)

    # Final norm + lm_head to get logits for last token
    h = norm(h)
    logits = lm_head(h)
    _gpu_sync(logits)

    prefill_time = time.perf_counter() - prefill_start
    prompt_tps = num_prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    logger.info(
        "SSD prefill done: %.2fs (%.1f tok/s)", prefill_time, prompt_tps
    )

    # ===================================================================
    # DECODE — per-layer weight recycling (same as prefill, 1 token)
    # ===================================================================
    generated_tokens: list[int] = []

    # Sample first token from prefill logits
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    token = sampler(logprobs)
    token_id = token.item()

    detokenizer = tokenizer.detokenizer
    decode_start = time.perf_counter()

    for n in range(max_tokens):
        # Check EOS
        if token_id in tokenizer.eos_token_ids:
            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token_id,
                logprobs=logprobs.squeeze(0),
                from_draft=False,
                prompt_tokens=num_prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / max(time.perf_counter() - decode_start, 1e-9),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop",
            )
            return

        detokenizer.add_token(token_id)
        generated_tokens.append(token_id)

        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token_id,
            logprobs=logprobs.squeeze(0),
            from_draft=False,
            prompt_tokens=num_prompt_tokens,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / max(time.perf_counter() - decode_start, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )

        # --- Decomposed forward pass for next token ---
        h = embed_tokens(mx.array([[token_id]]))
        mask = _create_attention_mask(h, prompt_cache)
        h = _per_layer_forward(h, mask)
        h = norm(h)
        logits = lm_head(h)
        logits = logits[:, -1, :]

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            penalty_tokens = mx.array(generated_tokens)
            penalty_logits = logits[0, penalty_tokens]
            penalty_logits = mx.where(
                penalty_logits > 0,
                penalty_logits / repetition_penalty,
                penalty_logits * repetition_penalty,
            )
            logits = logits.at[0, penalty_tokens].add(
                penalty_logits - logits[0, penalty_tokens]
            )

        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token = sampler(logprobs)
        token_id = token.item()

    # Max tokens reached
    detokenizer.add_token(token_id)
    detokenizer.finalize()
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=token_id,
        logprobs=logprobs.squeeze(0),
        from_draft=False,
        prompt_tokens=num_prompt_tokens,
        prompt_tps=prompt_tps,
        generation_tokens=max_tokens,
        generation_tps=max_tokens / max(time.perf_counter() - decode_start, 1e-9),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="length" if token_id not in tokenizer.eos_token_ids else "stop",
    )


def ssd_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, mx.array, list[int]],
    model_path: Union[str, Path],
    *,
    max_tokens: int = 256,
    **kwargs,
) -> str:
    """Non-streaming SSD disk-streaming generation.

    Collects all generated tokens and returns the full text.

    Args:
        model: The loaded MLX model.
        tokenizer: The tokenizer.
        prompt: Input text, token IDs, or mx.array.
        model_path: Path to model directory with safetensors files.
        max_tokens: Maximum tokens to generate.
        **kwargs: Additional arguments passed to ssd_stream_generate().

    Returns:
        The complete generated text string.
    """
    text_parts: list[str] = []
    for response in ssd_stream_generate(
        model, tokenizer, prompt, model_path, max_tokens=max_tokens, **kwargs
    ):
        text_parts.append(response.text)

    return "".join(text_parts)

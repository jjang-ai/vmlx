# SPDX-License-Identifier: Apache-2.0
"""
Weight index utilities for SSD disk-streaming inference.

Provides per-layer weight mapping, save/load/free operations that enable
recycling transformer layer weights from SSD during inference. This allows
models larger than physical RAM to run by loading only the active layer's
weights into Metal GPU memory and freeing them after use.

Key functions:
- build_weight_index: Map layer indices to safetensors files and weight keys
- save_layer_weights / save_all_layer_weights: Snapshot in-memory weights to disk
- load_layer_weights: Restore weights from safetensors into a layer
- free_layer_weights / free_all_layer_weights: Release Metal GPU memory
"""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

logger = logging.getLogger(__name__)

# MLX's mx.eval() is a GPU synchronization barrier — NOT Python's eval().
# It forces Metal to execute the pending compute graph synchronously.
_mx_eval = mx.eval  # noqa: S307 — this is mlx.core.eval, not builtins.eval

# Regex to extract layer index from weight keys.
# Matches patterns like:
#   model.layers.0.self_attn.q_proj.weight
#   language_model.model.layers.13.mlp.gate_proj.weight
#   model.language_model.layers.5.mlp.switch_mlp.gate_proj.weight
#   backbone.layers.0.linear_attn.in_proj_a.weight
_LAYER_RE = re.compile(
    r"^((?:model\.)?(?:language_model\.)?(?:model\.)?(?:backbone\.)?layers)\.(\d+)\."
)


def build_weight_index(model_path: str | Path) -> dict:
    """Map each transformer layer to its safetensors file(s) and weight keys.

    Args:
        model_path: Path to the model directory containing safetensors files.

    Returns:
        Dict mapping layer index (int) to:
            {"files": {filename: [key1, key2, ...]}, "prefix": "model.layers.N"}

    Raises:
        FileNotFoundError: If no safetensors files found.
    """
    model_path = Path(model_path)
    weight_map = _load_weight_map(model_path)

    if not weight_map:
        raise FileNotFoundError(
            f"No safetensors weight map found in {model_path}"
        )

    layers: dict[int, dict[str, Any]] = {}

    for key, filename in weight_map.items():
        match = _LAYER_RE.match(key)
        if not match:
            # Non-layer weight (embed_tokens, lm_head, norm, etc.) — skip
            continue

        prefix_base = match.group(1)  # e.g. "model.layers"
        layer_idx = int(match.group(2))

        if layer_idx not in layers:
            layers[layer_idx] = {
                "files": {},
                "prefix": f"{prefix_base}.{layer_idx}",
            }

        entry = layers[layer_idx]
        if filename not in entry["files"]:
            entry["files"][filename] = []
        entry["files"][filename].append(key)

    logger.debug(
        "Built weight index: %d layers from %s", len(layers), model_path
    )
    return layers


def save_layer_weights(
    model, layer_idx: int, output_dir: str | Path
) -> Path:
    """Save a single layer's in-memory weights to a safetensors file.

    Used for JANG models where on-disk weights don't match in-memory weights
    (JANG transforms weights during load: gate dequant, fc1/fc2 rename,
    expert restack).

    Args:
        model: The loaded model object.
        layer_idx: Index of the layer to save.
        output_dir: Directory to write the safetensors file.

    Returns:
        Path to the saved file.

    Raises:
        RuntimeError: If model layers cannot be found.
    """
    from vmlx_engine.utils.streaming_wrapper import _find_layers

    result = _find_layers(model)
    if result is None:
        raise RuntimeError("Cannot find transformer layers on model")

    container, attr_name = result
    all_layers = getattr(container, attr_name)
    layer = all_layers[layer_idx]

    # Flatten layer parameters to {relative_key: mx.array}
    tensors = _flatten_layer_params(layer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use mx.savez instead of safetensors — handles ALL MLX dtypes including
    # packed quantized uint arrays from QuantizedLinear (which safetensors rejects).
    file_path = output_dir / f"layer_{layer_idx:04d}.npz"
    mx.savez(str(file_path), **tensors)
    logger.debug(
        "Saved layer %d weights (%d tensors) to %s",
        layer_idx,
        len(tensors),
        file_path,
    )
    return file_path


def save_all_layer_weights(model, output_dir: str | Path) -> list[Path]:
    """Save ALL transformer layer weights to individual safetensors files.

    Used during JANG model init to create temp files for the load/unload cycle.

    Args:
        model: The loaded model object.
        output_dir: Directory to write the safetensors files.

    Returns:
        List of saved file paths, one per layer.

    Raises:
        RuntimeError: If model layers cannot be found.
    """
    from vmlx_engine.utils.streaming_wrapper import _find_layers

    result = _find_layers(model)
    if result is None:
        raise RuntimeError("Cannot find transformer layers on model")

    container, attr_name = result
    all_layers = getattr(container, attr_name)
    num_layers = len(all_layers)

    logger.info(
        "Saving %d layer weights to %s", num_layers, output_dir
    )

    paths = []
    for idx in range(num_layers):
        path = save_layer_weights(model, idx, output_dir)
        paths.append(path)
        if (idx + 1) % 10 == 0 or idx == num_layers - 1:
            logger.info("Saved %d/%d layer weights to SSD", idx + 1, num_layers)

    logger.info("Saved all %d layer weight files", len(paths))
    return paths


def load_layer_weights(
    layer,
    file_path: str | Path,
    weight_index_entry: dict | None = None,
) -> None:
    """Load weights from a safetensors file into a layer object.

    Handles two cases:
    1. Temp files (JANG): Keys are relative (no prefix) — load directly.
    2. Original model files: Keys have full prefix — strip before updating.

    Args:
        layer: The layer module to load weights into.
        file_path: Path to the safetensors file.
        weight_index_entry: Optional dict from build_weight_index with
            "prefix" and "files" info. If None, assumes temp file format
            (relative keys, no prefix stripping needed).
    """
    file_path = Path(file_path)
    raw_weights = mx.load(str(file_path))

    if weight_index_entry is None:
        # Temp file (JANG) — keys are already relative to the layer
        weights = _unflatten_params(raw_weights)
    else:
        # Original model file — extract only this layer's keys and strip prefix
        prefix = weight_index_entry["prefix"] + "."
        filename = file_path.name
        layer_keys = weight_index_entry["files"].get(filename, [])

        stripped = {}
        for key in layer_keys:
            if key in raw_weights:
                relative_key = key[len(prefix):]
                stripped[relative_key] = raw_weights[key]

        if not stripped:
            logger.warning(
                "No matching weights found in %s for prefix '%s'",
                file_path,
                prefix,
            )
            return

        weights = _unflatten_params(stripped)

    layer.update(weights)
    # Materialize weights into Metal GPU memory
    _mx_eval(layer.parameters())

    logger.debug("Loaded weights from %s", file_path)


def free_layer_weights(layer) -> None:
    """Replace all weight arrays in a layer with tiny placeholders.

    This frees Metal GPU memory by replacing every mx.array parameter with
    a single-element zero array. Experimentally confirmed to free memory:
    17,579MB -> 1,329MB.
    """
    params = layer.parameters()
    tiny = tree_map(
        lambda p: mx.zeros((1,)) if isinstance(p, mx.array) else p,
        params,
    )
    layer.update(tiny)
    # Force Metal to materialize the tiny placeholders (drops refs to old arrays)
    _mx_eval(layer.parameters())
    gc.collect()


def free_all_layer_weights(model) -> int:
    """Free ALL transformer layer weights to reclaim Metal GPU memory.

    More efficient than calling free_layer_weights N times — does one gc.collect
    at the end instead of per-layer.

    Args:
        model: The loaded model object.

    Returns:
        Number of layers freed, or 0 if layers not found.
    """
    from vmlx_engine.utils.streaming_wrapper import _find_layers

    result = _find_layers(model)
    if result is None:
        logger.warning("Cannot find transformer layers — nothing to free")
        return 0

    container, attr_name = result
    all_layers = getattr(container, attr_name)
    count = len(all_layers)

    for layer in all_layers:
        params = layer.parameters()
        tiny = tree_map(
            lambda p: mx.zeros((1,)) if isinstance(p, mx.array) else p,
            params,
        )
        layer.update(tiny)

    # Single eval + gc at the end — much faster than per-layer
    _mx_eval([l.parameters() for l in all_layers])
    gc.collect()

    logger.info("Freed weights for %d layers", count)
    return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_weight_map(model_path: Path) -> dict[str, str]:
    """Load weight-key-to-filename mapping from a model directory.

    Tries model.safetensors.index.json first (multi-file models), then
    falls back to listing safetensors files directly (single-file models).

    Returns:
        Dict mapping weight key -> safetensors filename.
    """
    index_path = model_path / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        logger.debug(
            "Loaded weight map from index: %d keys", len(weight_map)
        )
        return weight_map

    # Single-file model — build map by loading safetensors metadata
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        return {}

    weight_map = {}
    for st_file in st_files:
        try:
            tensors = mx.load(str(st_file))
            for key in tensors:
                weight_map[key] = st_file.name
        except Exception as e:
            logger.warning("Failed to read %s: %s", st_file, e)

    logger.debug(
        "Built weight map from %d file(s): %d keys",
        len(st_files),
        len(weight_map),
    )
    return weight_map


def _flatten_layer_params(layer) -> dict[str, mx.array]:
    """Flatten a layer's parameters to {dotted_key: mx.array}.

    Keys are relative to the layer (e.g. "self_attn.q_proj.weight",
    "mlp.gate_proj.scales") — no layer index prefix.
    """
    params = layer.parameters()
    flat = tree_flatten(params)
    return {k: v for k, v in flat if isinstance(v, mx.array)}


def _unflatten_params(flat_dict: dict[str, mx.array]) -> dict:
    """Convert a flat dotted-key dict back to a nested dict for layer.update().

    Example: {"self_attn.q_proj.weight": arr} ->
             {"self_attn": {"q_proj": {"weight": arr}}}
    """
    nested: dict = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested

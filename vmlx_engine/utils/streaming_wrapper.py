# SPDX-License-Identifier: Apache-2.0
"""
Layer-by-layer streaming wrapper for disk-streaming inference.

When models exceed physical RAM, MLX's lazy evaluation builds the entire
forward pass as a single compute graph. Metal tries to make ALL layer weights
GPU-resident simultaneously, causing OOM.

The fix: wrap each transformer layer so that mx.eval() is called after each
layer executes. This forces Metal to execute one layer at a time, allowing
macOS to page idle layer weights to SSD (~7.4GB/s) between layers.

Combined with a reduced wired memory limit (so Metal doesn't try to pin
all weights in non-pageable memory), this enables running models that
exceed RAM at ~2-5 tok/s.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

# MLX's mx.eval() is a GPU synchronization barrier — NOT Python's eval().
# It forces Metal to execute the pending compute graph synchronously.
# This is safe and required for layer-by-layer streaming.
_mx_eval = mx.eval  # noqa: S307 — this is mlx.core.eval, not builtins.eval


class StreamingLayerWrapper:
    """Wraps a transformer layer to force synchronous execution after it runs.

    After the wrapped layer executes, mx.eval() is called on the output
    hidden state. This breaks MLX's lazy graph at each layer boundary,
    forcing Metal to execute and free the current layer's buffers before
    the next layer loads.
    """

    def __init__(self, layer):
        self._layer = layer

    def __call__(self, *args, **kwargs):
        result = self._layer(*args, **kwargs)
        # Force synchronous GPU execution — breaks the lazy graph here.
        # This is the critical call that enables SSD paging: Metal executes
        # this layer's computation, then macOS can page out its weights
        # before the next layer loads them in.
        if isinstance(result, tuple):
            _mx_eval(*[r for r in result if isinstance(r, mx.array)])
        elif isinstance(result, mx.array):
            _mx_eval(result)
        else:
            # Some layers return custom objects; try to force-sync any arrays
            _mx_eval(result)
        return result

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _find_layers(model):
    """Find the transformer layers list on any supported model structure.

    Returns (layers_container, attr_name) so caller can do:
        setattr(container, attr_name, new_layers)

    The container must allow setattr — we avoid returning objects where
    'layers' is a read-only property (e.g., nemotron_h Model.layers is
    a property that delegates to backbone.layers).

    Supports:
    - model.model.layers — standard mlx-lm (llama, qwen, mistral, etc.)
    - model.language_model.model.layers — VLM wrapper (mistral3, pixtral, qwen-vl)
    - model.backbone.layers — nemotron_h (layers property delegates here)
    - model.model.layers with make_cache() — hybrid SSM
    """
    # VLM wrapper: model.language_model.model.layers
    lm = getattr(model, 'language_model', None)
    if lm is not None:
        inner = getattr(lm, 'model', None)
        if inner is not None and hasattr(inner, 'layers'):
            return inner, 'layers'

    # Standard mlx-lm: model.model.layers
    inner = getattr(model, 'model', None)
    if inner is not None and hasattr(inner, 'layers'):
        return inner, 'layers'

    # Nemotron-H / models with backbone: model.backbone.layers
    # (model.layers may be a property that delegates here — can't setattr on property)
    backbone = getattr(model, 'backbone', None)
    if backbone is not None and hasattr(backbone, 'layers'):
        return backbone, 'layers'

    # Direct layers on model — check it's not a read-only property
    if hasattr(model, 'layers'):
        # Check if 'layers' is a property on the class (can't setattr)
        for cls in type(model).__mro__:
            if 'layers' in cls.__dict__ and isinstance(cls.__dict__['layers'], property):
                logger.warning(f"model.layers is a read-only property on {cls.__name__} — cannot wrap")
                return None
        return model, 'layers'

    return None


def apply_streaming_layers(model):
    """Wrap each transformer layer with StreamingLayerWrapper.

    This is the main entry point. Call after model load when
    stream_from_disk=True.

    Returns the number of layers wrapped, or 0 if no layers found.
    """
    result = _find_layers(model)
    if result is None:
        logger.warning("Could not find model layers for streaming wrapper")
        return 0

    container, attr_name = result
    layers = getattr(container, attr_name)

    # Wrap each layer
    wrapped = [StreamingLayerWrapper(layer) for layer in layers]
    setattr(container, attr_name, wrapped)

    n = len(wrapped)
    logger.info(f"Streaming wrapper applied to {n} layers")
    return n


def compute_streaming_wired_limit(model, n_buffer_layers=3):
    """Compute a reduced wired memory limit for streaming mode.

    Instead of pinning ALL model weights in non-pageable (wired) memory,
    we only need enough for:
    - n_buffer_layers worth of weights (active + prefetching)
    - Embeddings + output head
    - KV cache headroom

    This tells macOS that weights beyond the wired limit CAN be paged to SSD.

    Returns the wired limit in bytes, or None if cannot compute.
    """
    result = _find_layers(model)
    if result is None:
        return None

    container, attr_name = result
    layers = getattr(container, attr_name)
    if not layers:
        return None

    # Estimate single layer size from the first layer's parameters
    layer = layers[0]
    # Unwrap if it's already a StreamingLayerWrapper
    if isinstance(layer, StreamingLayerWrapper):
        layer = layer._layer

    layer_bytes = 0
    try:
        from mlx.utils import tree_reduce
        layer_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
            layer.parameters(), 0,
        )
    except Exception:
        # Fallback: estimate from total model params
        try:
            from mlx.utils import tree_reduce
            total_bytes = tree_reduce(
                lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
                model.parameters(), 0,
            )
            n_layers = len(layers)
            layer_bytes = int(total_bytes * 0.9 / n_layers)
        except Exception:
            return None

    if layer_bytes == 0:
        return None

    # Wired limit = buffer layers + 50% headroom for embeddings/head/KV
    wired = int(layer_bytes * n_buffer_layers * 1.5)

    # Floor at 2GB to avoid starving the system
    wired = max(wired, 2 * 1024 * 1024 * 1024)

    # Cap at 50% of RAM — we want most memory to be pageable
    try:
        import psutil
        ram = psutil.virtual_memory().total
        wired = min(wired, int(ram * 0.5))
    except Exception:
        pass

    logger.info(
        f"Streaming wired limit: {wired / (1024**3):.1f}GB "
        f"(~{n_buffer_layers} layers x {layer_bytes / (1024**3):.2f}GB + headroom)"
    )
    return wired


# --- Wired limit lock ---
# mlx-lm's generate.py wraps every generate/stream_generate call in
# wired_limit_context() which sets wired limit to max. This defeats our
# reduced limit. We monkey-patch mx.set_wired_limit to be a no-op after
# we've set our streaming wired limit.

_original_set_wired_limit = None


def lock_wired_limit():
    """Prevent any code from changing the wired limit.

    Call this AFTER setting the streaming wired limit. It replaces
    mx.set_wired_limit with a no-op that returns the current value,
    so mlx-lm's wired_limit_context and BatchGenerator don't override it.
    """
    global _original_set_wired_limit
    if _original_set_wired_limit is not None:
        return  # Already locked

    _original_set_wired_limit = mx.set_wired_limit

    def _noop_set_wired_limit(limit):
        # Return 0 (the "previous" limit) without actually changing anything.
        # This satisfies callers that save the old limit for restoration.
        return 0

    mx.set_wired_limit = _noop_set_wired_limit
    logger.info("Wired limit locked — external set_wired_limit calls are no-ops")


def unlock_wired_limit():
    """Restore the original mx.set_wired_limit."""
    global _original_set_wired_limit
    if _original_set_wired_limit is not None:
        mx.set_wired_limit = _original_set_wired_limit
        _original_set_wired_limit = None
        logger.info("Wired limit unlocked")

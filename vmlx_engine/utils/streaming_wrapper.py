# SPDX-License-Identifier: Apache-2.0
"""
Model architecture utilities for SSD disk-streaming inference.

Provides _find_layers() to locate transformer layers across all supported
model architectures (standard mlx-lm, VLM, Nemotron-H, hybrid SSM).

The old StreamingLayerWrapper approach (mx.eval per layer + reduced wired limit)
has been replaced by per-layer weight recycling in ssd_generate.py and
weight_index.py. See docs/plans/2026-03-21-real-ssd-streaming-design.md.
"""

import logging

logger = logging.getLogger(__name__)


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

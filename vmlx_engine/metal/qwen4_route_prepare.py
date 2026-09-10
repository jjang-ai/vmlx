"""Opt-in Qwen routed-prefill permutation inversion without a second sort."""

import logging
import os

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

_ENABLED = os.environ.get("VMLX_QWEN4_SCATTER_ROUTE_INVERSE", "0") == "1"
_OBSERVED = False


def scatter_route_switchglu(switch, x, indices):
    """Keep stock sorted expert arithmetic; return None outside the opt-in lane."""
    if (not _ENABLED or type(switch) is not SwitchGLU or switch.training
            or x.ndim not in (2, 3) or indices.size < 64
            or indices.ndim != x.ndim
            or indices.shape[:-1] != x.shape[:-1]
            or indices.dtype not in (mx.int32, mx.uint32)):
        return None
    global _OBSERVED
    if not _OBSERVED:
        logging.getLogger(__name__).info(
            "Qwen scatter route inverse active: rows=%d top_k=%d",
            indices.size // indices.shape[-1], indices.shape[-1],
        )
        _OBSERVED = True
    flat = indices.flatten()
    order = mx.argsort(flat)
    # argsort returns a permutation: every output index is written exactly once.
    # Integer scatter-add therefore has no collision/reduction-order ambiguity.
    inverse = mx.zeros(order.shape, dtype=order.dtype).at[order].add(
        mx.arange(order.size, dtype=order.dtype)
    )
    value = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[
        order // indices.shape[-1]
    ]
    selected = flat[order]
    up = switch.up_proj(value, selected, sorted_indices=True)
    gate = switch.gate_proj(value, selected, sorted_indices=True)
    result = switch.down_proj(
        switch.activation(up, gate), selected, sorted_indices=True
    )
    return mx.unflatten(result[inverse], 0, indices.shape).squeeze(-2)

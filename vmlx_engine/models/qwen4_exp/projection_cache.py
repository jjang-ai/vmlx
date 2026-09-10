"""Opt-in guarded reuse of validated Qwen affine projection groups."""

import logging
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.metal.quantized_projection_group import (
    QuantizedProjectionGroup,
    quantized_projection_group_reason,
)


def validated_projection_group(linears, activation_dtype):
    # Only standard MLX modules have the dictionary-backed tensor contract.
    # Subclasses may override attribute access and retain the existing caller path.
    if not linears or any(type(m) is not nn.QuantizedLinear for m in linears):
        return None
    parts = [activation_dtype]
    tensors = []
    for module in linears:
        weight = module.get("weight")
        scales = module.get("scales")
        biases = module.get("biases")
        if not all(isinstance(a, mx.array) for a in (weight, scales, biases)):
            return None
        parts.append((
            id(module), module.bits, module.group_size, module.mode,
            "bias" in module,
            id(weight), weight.shape, weight.dtype,
            id(scales), scales.shape, scales.dtype,
            id(biases), biases.shape, biases.dtype,
        ))
        tensors.extend((weight, scales, biases))
    signature = tuple(parts)
    owner = linears[0]
    cached = owner.get("_qwen4_validated_projection_group")
    if cached is not None and cached[0] == signature:
        return cached[1]
    if quantized_projection_group_reason(
        linears, activation_dtype=activation_dtype
    ) is not None:
        return None
    group = QuantizedProjectionGroup(linears)
    mx.eval(group.weight, group.scales, group.biases)
    owner._qwen4_validated_projection_group = (signature, group)
    # Keep original objects alive so a recycled Python id cannot masquerade as
    # an unchanged tensor. Do not expose duplicate source tensors as parameters.
    owner._qwen4_projection_source_refs = SimpleNamespace(tensors=tuple(tensors))
    logging.getLogger(__name__).info(
        "Qwen guarded projection group prepared: projections=%d bits=%s "
        "group_size=%s activation_dtype=%s",
        len(linears), linears[0].bits, linears[0].group_size, activation_dtype,
    )
    return group

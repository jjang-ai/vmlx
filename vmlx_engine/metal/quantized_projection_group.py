"""Exact grouping for compatible affine ``QuantizedLinear`` projections.

Affine quantization metadata is independent for every output row. Compatible
same-input projections can therefore be concatenated on the output-row axis
and evaluated by one ``mx.quantized_matmul`` without changing their math. The
grouped call retains MLX's normal device/shape dispatcher while removing
redundant command launches. Backend selection (including any M5 TensorOps/NAX
path) remains a separate live-measurement question.

This module does not choose model policy. Callers decide whether to retain the
original projections for a diagnostic cache or replace them at load time to
keep resident memory neutral.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn


def _quantized_linear_dimensions(linear: nn.QuantizedLinear) -> tuple[int, int]:
    """Derive logical dimensions from MLX's stored quantization tensors."""

    output_dims = int(linear.weight.shape[0])
    input_dims = int(linear.scales.shape[-1]) * int(linear.group_size)
    return input_dims, output_dims


class QuantizedProjectionGroup(nn.Module):
    """One packed affine projection that returns the original output tuple."""

    def __init__(self, linears: Sequence[nn.Module]):
        super().__init__()
        linears = tuple(linears)
        reason = quantized_projection_group_reason(linears)
        if reason is not None:
            raise ValueError(reason)

        first = linears[0]
        self.group_size = int(first.group_size)
        self.bits = int(first.bits)
        self.mode = str(first.mode)
        self.input_dims, _ = _quantized_linear_dimensions(first)
        self.output_dims = sum(
            _quantized_linear_dimensions(linear)[1] for linear in linears
        )
        self.weight = mx.concatenate([linear.weight for linear in linears], axis=0)
        self.scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        # MX modes (mxfp8/mxfp4/nvfp4) carry no biases; their scales are per output row too, so row
        # concatenation is exact exactly as for affine (JANGTQ v2 campaign generalization).
        self.biases = (mx.concatenate([linear.biases for linear in linears], axis=0)
                       if self.mode == "affine" else None)

        splits: list[int] = []
        offset = 0
        for linear in linears[:-1]:
            offset += _quantized_linear_dimensions(linear)[1]
            splits.append(offset)
        self.split_indices = tuple(splits)
        self.freeze()

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        output = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return tuple(mx.split(output, self.split_indices, axis=-1))

    def separate(self, x: mx.array) -> tuple[mx.array, ...]:
        """Run original-size QMMs from non-owning row slices of grouped storage.

        Large concatenated output matrices can select a different MLX prefill
        kernel than the original projections.  Calling each row slice
        separately preserves that original dispatch and its exact bytes while
        keeping only one resident copy of the packed weights and metadata.
        """

        boundaries = (0, *self.split_indices, self.output_dims)
        return tuple(
            mx.quantized_matmul(
                x,
                self.weight[start:end],
                scales=self.scales[start:end],
                biases=(self.biases[start:end] if self.biases is not None else None),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            for start, end in zip(boundaries, boundaries[1:])
        )


def quantized_projection_group_reason(
    linears: Sequence[nn.Module],
    *,
    activation_dtype: mx.Dtype | None = None,
) -> str | None:
    """Return why projections cannot share one exact affine dispatch."""

    linears = tuple(linears)
    if not linears:
        return "no projections"
    if not all(isinstance(linear, nn.QuantizedLinear) for linear in linears):
        return "projection is not QuantizedLinear"
    if any(hasattr(linear, "hadamard_block") for linear in linears):
        # Packing raw rows would bypass the per-module activation transform.
        # Keep these callable projections intact; equal shape/bits is not
        # sufficient proof that their transformed inputs are interchangeable.
        return "projection requires a Hadamard activation transform"

    first = linears[0]
    first_input_dims, _ = _quantized_linear_dimensions(first)
    mx_modes = ("mxfp8", "mxfp4", "nvfp4")
    is_affine = str(first.mode) == "affine"
    if not is_affine and str(first.mode) not in mx_modes:
        return f"unsupported quantization mode {first.mode}"
    for linear in linears:
        biases = getattr(linear, "biases", None)
        if linear.weight.ndim != 2:
            return "packed weight is not rank two"
        if linear.scales.ndim != 2:
            return "scales are not rank two"
        if is_affine:
            if biases is None:
                return "affine biases are missing"
            if biases.shape != linear.scales.shape:
                return "affine metadata shapes differ"
            if linear.weight.shape[0] != biases.shape[0]:
                return "output-row counts differ within a projection"
        elif biases is not None:
            return "MX-mode projection unexpectedly carries biases"
        if linear.weight.shape[0] != linear.scales.shape[0]:
            return "output-row counts differ within a projection"
        input_dims, _ = _quantized_linear_dimensions(linear)
        if int(linear.weight.shape[-1]) * 32 != input_dims * int(linear.bits):
            return "packed weight geometry is inconsistent"
        if input_dims != first_input_dims:
            return "input dimensions differ"
        if int(linear.bits) != int(first.bits):
            return "bit widths differ"
        if int(linear.group_size) != int(first.group_size):
            return "group sizes differ"
        if str(linear.mode) != str(first.mode):
            return "quantization modes differ"
        if "bias" in linear:
            return "post-matmul bias is unsupported"
        if linear.weight.dtype != first.weight.dtype:
            return "packed weight dtypes differ"
        if linear.scales.dtype != first.scales.dtype:
            return "scale dtypes differ"
        if is_affine and linear.biases.dtype != first.biases.dtype:
            return "affine bias dtypes differ"
    if is_affine and activation_dtype is not None and (
        first.scales.dtype != activation_dtype
        or first.biases.dtype != activation_dtype
    ):
        return "activation and affine metadata dtypes differ"
    return None


def cached_quantized_projection_group(
    linears: Sequence[nn.Module],
    *,
    owner: object,
    cache_attr: str,
) -> QuantizedProjectionGroup:
    """Build and cache a group while retaining the caller's projections."""

    linears = tuple(linears)
    source_key = tuple(
        (id(linear.weight), id(linear.scales), id(getattr(linear, "biases", None)))
        for linear in linears
    )
    cached = getattr(owner, cache_attr, None)
    if cached is None or cached[0] != source_key:
        group = QuantizedProjectionGroup(linears)
        mx.eval(
            group.weight, group.scales,
            *(() if group.biases is None else (group.biases,)),
        )
        cached = (source_key, group)
        setattr(owner, cache_attr, cached)
    return cached[1]


__all__ = [
    "QuantizedProjectionGroup",
    "cached_quantized_projection_group",
    "quantized_projection_group_reason",
]

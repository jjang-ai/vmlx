# SPDX-License-Identifier: Apache-2.0
"""Preserve optional saved parameters omitted by newer native image modules."""
import logging

logger = logging.getLogger(__name__)


def restore_output_modulation_bias(transformer, stored_parameters):
    """Restore an exporter's real output bias, never invent one for bias-free saves.

    Older Flux1/Qwen exports used bias=True for this projection. Current mflux
    constructs it with bias=False and permissive Module.update discards the
    saved bias. Adding the original tensor also works on QuantizedLinear: its
    packed weights, affine scales and per-layer quantization remain untouched.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    bias = dict(tree_flatten(stored_parameters)).get('norm_out.linear.bias')
    if bias is None:
        return False
    linear = transformer.norm_out.linear
    if type(linear) not in (nn.Linear, nn.QuantizedLinear):
        raise ValueError('Unsupported saved output modulation projection')
    if (bias.shape != (linear.weight.shape[0],)
            or not mx.issubdtype(bias.dtype, mx.floating)):
        raise ValueError('Invalid saved output modulation bias shape or dtype')
    # No parameter publication before validation; no casting or requantization.
    linear.bias = bias
    logger.info('Restored saved output modulation bias (%d values); original projection precision retained', bias.size)
    return True

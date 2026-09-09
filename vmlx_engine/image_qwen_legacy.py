# SPDX-License-Identifier: Apache-2.0
"""Value-preserving mflux 0.11 Qwen modulation layout compatibility."""
import logging

logger = logging.getLogger(__name__)


def restore_legacy_modulation(transformer, stored_parameters):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    stored = dict(tree_flatten(stored_parameters))
    replacements = []
    for index, block in enumerate(transformer.transformer_blocks):
        for lane in ('img', 'txt'):
            old = f'transformer_blocks.{index}.{lane}_norm1.mod_linear.'
            new = f'transformer_blocks.{index}.{lane}_mod_linear.'
            values = {key[len(old):]: value for key, value in stored.items() if key.startswith(old)}
            if not values:
                continue
            if any(key.startswith(new) for key in stored):
                raise ValueError(f'Ambiguous old/new Qwen modulation weights: {new}')
            original = getattr(block, f'{lane}_mod_linear')
            if type(original) is not nn.Linear:
                raise ValueError(f'Unexpected Qwen modulation topology: {new}')
            out_dims, in_dims = original.weight.shape
            weight = values.get('weight')
            bias = values.get('bias')
            if weight is None or weight.ndim != 2 or weight.shape[0] != out_dims:
                raise ValueError(f'Invalid Qwen modulation weight: {old}')
            if bias is None or bias.shape != (out_dims,):
                raise ValueError(f'Missing or invalid Qwen modulation bias: {old}')
            if weight.dtype == mx.uint32:
                scales, biases = values.get('scales'), values.get('biases')
                if (scales is None or biases is None or scales.ndim != 2
                        or scales.shape != biases.shape or scales.shape[0] != out_dims
                        or scales.shape[-1] == 0 or in_dims % scales.shape[-1]
                        or weight.shape[-1] * 32 % in_dims):
                    raise ValueError(f'Invalid packed Qwen modulation layout: {old}')
                bits, group = weight.shape[-1]*32//in_dims, in_dims//scales.shape[-1]
                if bits not in (2,3,4,5,6,8) or group not in (32,64,128):
                    raise ValueError(f'Unsupported packed Qwen modulation layout: {old}')
                if not all(mx.issubdtype(v.dtype, mx.floating) for v in (scales,biases,bias)):
                    raise ValueError(f'Invalid Qwen modulation affine dtype: {old}')
                linear = nn.QuantizedLinear(in_dims,out_dims,bias=True,bits=bits,group_size=group)
                expected = {'weight','scales','biases','bias'}
            else:
                if weight.shape != original.weight.shape or not mx.issubdtype(weight.dtype,mx.floating):
                    raise ValueError(f'Invalid unquantized Qwen modulation layout: {old}')
                linear = nn.Linear(in_dims,out_dims,bias=True)
                expected = {'weight','bias'}
            if set(values) != expected:
                raise ValueError(f'Unexpected Qwen modulation parameters: {old}')
            linear.update(values,strict=True)
            replacements.append((block,f'{lane}_mod_linear',linear))

    if not replacements:
        return 0
    # mflux 0.11's AdaLayerNormContinuous used bias=True. Current shared Flux
    # module uses bias=False; Qwen's saved output bias must not be discarded.
    output_bias = stored.get('norm_out.linear.bias')
    if output_bias is None or output_bias.shape != (transformer.norm_out.embedding_dim * 2,):
        raise ValueError('Missing or invalid legacy Qwen output modulation bias')
    for block, name, linear in replacements:
        setattr(block,name,linear)
    transformer.norm_out.linear.bias = output_bias
    logger.info('Restored %d legacy Qwen modulation layers and saved output bias; original packed tensors retained',len(replacements))
    return len(replacements)

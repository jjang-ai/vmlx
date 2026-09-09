# SPDX-License-Identifier: Apache-2.0
"""Restore already-packed legacy image encoder layers, never quantize floats.

Recent mflux Qwen definitions skip encoder quantization for quality. Legacy
mflux saves can nevertheless contain quantized encoders. Loading those arrays
into ordinary Embedding/Linear modules leaves packed columns as activations.
Use a fresh native topology to recover dimensions, then attach the original
packed weights/scales/biases at their stored per-layer precision.
"""
import logging

logger = logging.getLogger(__name__)


def restore_packed_encoder(encoder, load_stored):
    """Return (encoder, restored_count); unquantized/already-quantized is no-op.

    The caller must supply an encoder with a no-argument native constructor.
    load_stored lazily returns the original component tree: Module.update on
    an ordinary layer has already discarded unfamiliar scales/biases keys.
    No disk writes, global dependency patches, dtype casts or model-name bits.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    broken = [name for name, module in encoder.named_modules()
              if type(module) in (nn.Embedding, nn.Linear)
              and module.weight.dtype == mx.uint32]
    if not broken:
        return encoder, 0

    stored_parameters = load_stored()
    stored = dict(tree_flatten(stored_parameters))
    native = type(encoder)()
    plans = {}
    for name, module in native.named_modules():
        if type(module) not in (nn.Embedding, nn.Linear):
            continue
        weight = stored.get(f"{name}.weight")
        if weight is None:
            raise ValueError(f"Missing stored encoder weight: {name}")
        if weight.dtype != mx.uint32:
            if weight.shape != module.weight.shape:
                raise ValueError(f"Stored encoder shape mismatch: {name}")
            continue
        scales, biases = (stored.get(f"{name}.{key}") for key in ('scales', 'biases'))
        dims = module.weight.shape[-1]
        if (weight.ndim != 2 or scales is None or biases is None
                or scales.ndim != 2 or biases.shape != scales.shape
                or weight.shape[0] != module.weight.shape[0]
                or scales.shape[0] != weight.shape[0] or scales.shape[-1] == 0):
            raise ValueError(f"Incomplete packed encoder layer: {name}")
        packed_bits = weight.shape[-1] * 32
        if packed_bits % dims or dims % scales.shape[-1]:
            raise ValueError(f"Invalid packed encoder dimensions: {name}")
        bits, group_size = packed_bits // dims, dims // scales.shape[-1]
        if bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
            raise ValueError(f"Unsupported stored encoder quantization: {name}, {bits}bit/group{group_size}")
        if not mx.issubdtype(scales.dtype, mx.floating) or not mx.issubdtype(biases.dtype, mx.floating):
            raise ValueError(f"Invalid packed encoder scale dtype: {name}")
        plans[name] = {'bits': bits, 'group_size': group_size}

    if not set(broken).issubset(plans):
        raise ValueError("Stored encoder topology does not match the native adapter")
    nn.quantize(native, class_predicate=lambda path, module: plans.get(path, False))
    native.update(stored_parameters, strict=True)
    logger.info("Restored %d legacy packed image encoder layers from stored shapes; layouts=%s; tensor values unchanged",
                len(plans), sorted({(v['bits'], v['group_size']) for v in plans.values()}))
    return native, len(plans)

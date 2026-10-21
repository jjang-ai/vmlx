"""Load stored affine DFlash2 checkpoints without changing their quantization.

The upstream loader handles floating checkpoints. Packed checkpoints need
quantized modules installed before strict weight loading; floating selector,
normalization and convolution tensors remain exactly as stored.
"""

import json
from dataclasses import fields
from pathlib import Path


def load_quantized_dflash2(path):
    import dflash.model_mlx as runtime
    import mlx.core as mx
    import mlx.nn as nn

    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("mode", "affine") != "affine":
        raise ValueError("DFlash2 stored quantization must declare affine bits and group_size")
    bits, group_size = quant.get("bits"), quant.get("group_size")
    if bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
        raise ValueError("Unsupported DFlash2 stored affine quantization geometry")

    dflash = cfg.get("dflash_config", {})
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    # Keep the upstream configuration defaults and trained geometry. These
    # fields describe the model, not the adaptive verifier's live draft depth.
    values = {f.name: cfg[f.name] for f in fields(runtime.DFlashConfig) if f.name in cfg}
    values.update({
        "rope_theta": cfg.get("rope_theta", (rope or {}).get("rope_theta", 10000.0)),
        "rope_scaling": rope,
        "layer_types": tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]),
        "block_size": int(dflash.get("block_size", cfg.get("block_size", 16))),
        "target_layer_ids": tuple(dflash["target_layer_ids"]),
        "mask_token_id": dflash["mask_token_id"],
    })
    for name in ("final_logit_softcapping", "input_embedding_scale", "output_multiplier",
                 "conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
        if name in dflash:
            values[name] = dflash[name]
    if len(values["layer_types"]) != cfg["num_hidden_layers"]:
        raise ValueError("Draft config layer_types length must match num_hidden_layers")
    if set(values["layer_types"]) - {"full_attention", "sliding_attention"}:
        raise ValueError("Unsupported DFlash2 draft layer type")
    if "sliding_attention" in values["layer_types"] and cfg.get("sliding_window") is None:
        raise ValueError("Sliding attention draft requires sliding_window")

    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        if shard.name.startswith("._"):
            continue
        current = mx.load(str(shard))
        if weights.keys() & current.keys():
            raise ValueError("Duplicate DFlash2 tensor across shards")
        weights.update(current)
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        if key in weights:
            if f"{key}.weight" in weights:
                raise ValueError(f"Duplicate DFlash2 codebook: {key}")
            weights[f"{key}.weight"] = weights.pop(key)
    if not any(key.endswith(".scales") for key in weights):
        raise ValueError("DFlash2 quantization declared without packed tensor scales")
    model = runtime.DFlash2DraftModel(runtime.DFlashConfig(**values))
    nn.quantize(model, bits=bits, group_size=group_size,
                class_predicate=lambda name, module: f"{name}.scales" in weights)
    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model

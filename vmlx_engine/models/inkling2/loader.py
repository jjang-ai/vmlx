"""Load Inkling Mini's indexed target weights without reading draft tensors."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .fused_nvfp4_experts import prepare_fused_nvfp4_experts
from .inkling2 import Model
from .nvfp4 import replace_scaled_quantized_linears
from .processing_inkling2 import Inkling2Processor


logger = logging.getLogger(__name__)


def register_inkling2_runtime() -> None:
    from mlx_vlm import models, prompt_utils

    module = sys.modules[__package__]
    sys.modules["mlx_vlm.models.inkling2_loop"] = module
    models.inkling2_loop = module
    prompt_utils.MODEL_CONFIG["inkling2_loop"] = (
        prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST
    )


def load_native_inkling2(path: Path, config: dict):
    if "global_scale" not in (mx.gather_qmm.__doc__ or ""):
        raise RuntimeError("Inkling Mini NVFP4 requires MLX 0.32.4 or newer")

    model = Model(ModelConfig.from_dict(config))
    quantization = config.get("quantization")
    if not quantization:
        metadata = json.loads((path / "hf_quant_config.json").read_text())
        quantization = model.quantization_from_metadata(metadata)
    if not quantization:
        raise ValueError("Inkling Mini is missing its native quantization layout")

    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = {}
    for name in sorted(set(index["weight_map"].values())):
        shard = Path(name)
        if shard.name != name or shard.suffix != ".safetensors":
            raise ValueError(f"Invalid Inkling target shard name: {name!r}")
        weights.update(mx.load(path / shard))
    weights = model.sanitize(weights)

    def should_quantize(name, module):
        if hasattr(module, "to_quantized") and f"{name}.scales" in weights:
            return quantization.get(name, True)
        return False

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization["mode"],
        class_predicate=should_quantize,
    )
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    expert_modes = {
        getattr(projection, "mode", None)
        for layer in model.language_model.model.layers
        for projection in (
            layer.mlp.switch_mlp.gate_proj,
            layer.mlp.switch_mlp.up_proj,
            layer.mlp.switch_mlp.down_proj,
        )
    }
    model._inkling_loaded_expert_modes = tuple(
        mode for mode in ("nvfp4", "mxfp8") if mode in expert_modes
    )
    fused_layers = prepare_fused_nvfp4_experts(model.language_model)
    if fused_layers:
        logger.info("Inkling NVFP4 fused expert decode enabled for %d layers", fused_layers)
    controllers = sum(
        controller.prepare_fused_decode()
        for layer in model.language_model.model.layers
        for controller in getattr(layer, "controllers", {}).values()
    )
    if controllers:
        logger.info("Inkling fused loop decode enabled for %d controllers", controllers)
    return model, Inkling2Processor.from_pretrained(path)

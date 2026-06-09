# SPDX-License-Identifier: Apache-2.0
"""
MLX Multimodal Language Model (MLLM) wrapper.

This module provides a wrapper around mlx-vlm for multimodal inference,
supporting vision, audio, and video understanding on Apple Silicon.

Features:
- OpenAI-compatible API format for images and video
- Smart video frame extraction with configurable FPS
- Base64 and URL image support
- Streaming generation
- MLLM KV cache for repeated image/video+prompt combinations
"""

import atexit
import base64
import importlib
import importlib.machinery
import json
import logging
import math
import os
import sys
import tempfile
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import numpy as np
import requests

from vmlx_engine.mlx_memory import clear_mlx_memory_cache
from vmlx_engine.mllm_cache import MLLMPrefixCacheManager
from vmlx_engine.errors import UnsupportedMediaModalityError

logger = logging.getLogger(__name__)


_VLM_STREAM = None


def _register_local_mlx_vlm_runtime_if_needed(model_path: str | Path) -> None:
    """Register vMLX-owned mlx-vlm model classes before stock mlx-vlm import.

    Some locally converted bundles use architectures that are not yet shipped
    by upstream mlx-vlm. Stock `mlx_vlm.utils.load()` resolves the model class
    from `config.json["model_type"]` before weights are inspected, so local
    adapters must be registered at the MLLM load boundary. This is not a
    monkey-patch of an instantiated model; it supplies the missing model module
    that mlx-vlm's normal loader contract already expects.
    """
    try:
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.exists():
            return
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type == "zaya1_vl":
        from ..models.zaya1_vl import register_mlx_vlm_zaya1_vl

        register_mlx_vlm_zaya1_vl()
    if model_type == "step3p7":
        _register_step3p7_mlx_vlm_runtime()
    if model_type == "mimo_v2":
        _register_mimo_v2_mlx_vlm_runtime()


def _register_step3p7_mlx_vlm_runtime() -> None:
    module_name = "mlx_vlm.models.step3p7"
    processor_module_name = f"{module_name}.processing_step3"
    if module_name in sys.modules and processor_module_name in sys.modules:
        return

    source_runtime = importlib.import_module("vmlx_engine.models.step3p7_mlx_vlm")
    module = types.ModuleType(module_name)
    for name in (
        "Model",
        "ModelConfig",
        "TextConfig",
        "VisionConfig",
        "ProjectorConfig",
        "LanguageModel",
        "VisionModel",
        "Step3p7Projector",
        "Step3VisionProcessor",
        "Step3VLProcessor",
    ):
        setattr(module, name, getattr(source_runtime, name))
    module.__file__ = getattr(source_runtime, "__file__", module_name)
    module.__package__ = module_name
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name,
        loader=None,
        origin=module.__file__,
        is_package=True,
    )
    sys.modules[module_name] = module

    processor_module = types.ModuleType(processor_module_name)
    processor_module.Step3VisionProcessor = source_runtime.Step3VisionProcessor
    processor_module.Step3VLProcessor = source_runtime.Step3VLProcessor
    processor_module.__file__ = getattr(source_runtime, "__file__", processor_module_name)
    processor_module.__package__ = module_name
    processor_module.__spec__ = importlib.machinery.ModuleSpec(
        processor_module_name,
        loader=None,
        origin=processor_module.__file__,
        is_package=False,
    )
    sys.modules[processor_module_name] = processor_module


def _install_mimo_v2_compiled_router_if_missing(text_runtime: Any) -> bool:
    """Install the MiMo single-token compiled router when jang-tools is stale.

    Newer `jang_tools.mimo_v2.mlx_model` ships this optimization natively.
    vMLX still has to tolerate older bundled environments because the app
    resolves MiMo through the bundled Python package at runtime. This patch is
    deliberately narrow: it only replaces the router's `__call__`, keeps the
    model's selected experts/weights semantics unchanged, and falls back to the
    original implementation for non-decode/batched shapes or compile failures.
    """

    if hasattr(text_runtime, "run_compiled_mimo_decode_router"):
        return False

    gate_cls = getattr(text_runtime, "MiMoV2MoEGate", None)
    if gate_cls is None or getattr(gate_cls, "_vmlx_compiled_router_patched", False):
        return False

    try:
        import mlx.core as mx
    except Exception:
        return False

    original_call = gate_cls.__call__
    router_cache: dict[tuple[int, bool, int], Any] = {}

    def _compiled_router(top_k: int, norm_topk_prob: bool, routed_scaling: float):
        scaling_milli = int(round(float(routed_scaling) * 1000.0))
        key = (int(top_k), bool(norm_topk_prob), scaling_milli)
        cached = router_cache.get(key)
        if cached is not None:
            return cached

        def _router(x_fp32, weight_fp32, bias_fp32):
            logits = x_fp32 @ weight_fp32.T
            scores = mx.sigmoid(logits)
            corrected = scores + bias_fp32.reshape(1, -1)
            inds = mx.argpartition(-corrected, kth=top_k - 1, axis=-1)[..., :top_k]
            weights = mx.take_along_axis(scores, inds, axis=-1)
            if norm_topk_prob:
                weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
            return inds, weights * (float(scaling_milli) * 0.001)

        compiled = mx.compile(_router)
        router_cache[key] = compiled
        return compiled

    def _vmlx_compiled_router_call(self, x):
        try:
            if int(x.shape[0]) != 1:
                return original_call(self, x)
            x_fp32 = x.astype(mx.float32)
            weight_source_id = id(self.weight)
            weight_fp32 = getattr(self, "_vmlx_gate_weight_fp32", None)
            if (
                weight_fp32 is None
                or getattr(self, "_vmlx_gate_weight_source_id", None) != weight_source_id
            ):
                weight_fp32 = self.weight.astype(mx.float32)
                self._vmlx_gate_weight_fp32 = weight_fp32
                self._vmlx_gate_weight_source_id = weight_source_id
            bias_source_id = id(self.e_score_correction_bias)
            bias_fp32 = getattr(self, "_vmlx_gate_bias_fp32", None)
            if (
                bias_fp32 is None
                or getattr(self, "_vmlx_gate_bias_source_id", None) != bias_source_id
            ):
                bias_fp32 = self.e_score_correction_bias.astype(mx.float32)
                self._vmlx_gate_bias_fp32 = bias_fp32
                self._vmlx_gate_bias_source_id = bias_source_id
            router = _compiled_router(
                int(self.top_k),
                bool(self.norm_topk_prob),
                float(self.routed_scaling),
            )
            topk_indices, topk_weights = router(
                x_fp32,
                weight_fp32,
                bias_fp32,
            )
            return topk_indices, topk_weights.astype(x.dtype)
        except Exception:
            return original_call(self, x)

    gate_cls._vmlx_original_call = original_call
    gate_cls.__call__ = _vmlx_compiled_router_call
    gate_cls._vmlx_compiled_router_patched = True
    logger.info("Installed vMLX fallback compiled MiMo-V2 decode router")
    return True


def _install_mimo_v2_affine_switchglu_decode_fast_path() -> bool:
    """Install an affine QuantizedSwitchLinear decode fast path for MiMo.

    Stock `SwitchGLU.__call__` runs routed gate/up/down through three separate
    `QuantizedSwitchLinear` module calls. On MiMo-V2 decode that is 47 MoE
    layers * 3 gather_qmm dispatches per token, which matches the observed
    ~600 ms/token even when the actual model graph is quantized. This path
    keeps the same MLX affine `gather_qmm` math, but compiles the decode-shape
    gate/up/SwiGLU/down sequence as one callable per shape.
    """

    try:
        from mlx_lm.models.switch_layers import SwitchGLU
        import mlx.core as mx
    except Exception:
        return False

    try:
        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
        from jang_tools.turboquant.fused_gate_up_kernel import make_fused_gate_up_swiglu_decode
        from jang_tools.turboquant.gather_tq_kernel import make_gather_tq_decode_per_row
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal
    except Exception:
        TurboQuantSwitchLinear = None
        make_fused_gate_up_swiglu_decode = None
        make_gather_tq_decode_per_row = None
        hadamard_rotate_metal = None

    if getattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_path", False):
        return False

    original_call = SwitchGLU.__call__
    compiled: dict[tuple[Any, ...], Any] = {}
    tq_compiled: dict[tuple[Any, ...], Any] = {}

    def _record_fallback(reason: str) -> None:
        try:
            reasons = SwitchGLU._vmlx_mimo_affine_decode_fallback_reasons
            reasons[reason] = int(reasons.get(reason, 0)) + 1
        except Exception:
            pass

    def _compiled_for(gp, up, dp, k: int):
        key = (
            int(gp.input_dims),
            int(gp.output_dims),
            int(getattr(gp, "bits", 0) or 0),
            int(getattr(gp, "group_size", 0) or 0),
            str(getattr(gp, "mode", "affine")),
            int(getattr(up, "bits", 0) or 0),
            int(getattr(up, "group_size", 0) or 0),
            str(getattr(up, "mode", "affine")),
            int(getattr(dp, "bits", 0) or 0),
            int(getattr(dp, "group_size", 0) or 0),
            str(getattr(dp, "mode", "affine")),
            int(k),
        )
        fn = compiled.get(key)
        if fn is not None:
            return fn

        def _mlp(
            x_flat,
            indices_flat,
            gate_w,
            gate_s,
            gate_b,
            up_w,
            up_s,
            up_b,
            down_w,
            down_s,
            down_b,
        ):
            x_exp = mx.expand_dims(x_flat, (-2, -3))
            idx = indices_flat.reshape(1, -1)
            x_up = mx.gather_qmm(
                x_exp,
                up_w,
                up_s,
                up_b,
                rhs_indices=idx,
                transpose=True,
                group_size=up.group_size,
                bits=up.bits,
                mode=up.mode,
                sorted_indices=False,
            )
            x_gate = mx.gather_qmm(
                x_exp,
                gate_w,
                gate_s,
                gate_b,
                rhs_indices=idx,
                transpose=True,
                group_size=gp.group_size,
                bits=gp.bits,
                mode=gp.mode,
                sorted_indices=False,
            )
            x_act = (x_gate * mx.sigmoid(x_gate)) * x_up
            x_out = mx.gather_qmm(
                x_act,
                down_w,
                down_s,
                down_b,
                rhs_indices=idx,
                transpose=True,
                group_size=dp.group_size,
                bits=dp.bits,
                mode=dp.mode,
                sorted_indices=False,
            )
            return x_out.squeeze(-2)

        fn = mx.compile(_mlp)
        compiled[key] = fn
        return fn

    def _compiled_tq_for(gp, up, dp, k: int):
        key = (
            int(gp.in_features),
            int(gp.out_features),
            int(getattr(gp, "bits", 0) or 0),
            int(getattr(up, "bits", 0) or 0),
            int(getattr(dp, "bits", 0) or 0),
            int(k),
        )
        fn = tq_compiled.get(key)
        if fn is not None:
            return fn

        fused_gate_up = make_fused_gate_up_swiglu_decode(
            gp.in_features,
            gp.out_features,
            gp.bits,
            k,
            swiglu_limit=0.0,
        )
        gather_down = make_gather_tq_decode_per_row(
            gp.out_features,
            gp.in_features,
            dp.bits,
            k,
        )

        def _mlp(
            x_flat,
            indices_flat,
            gate_packed,
            gate_norms,
            up_packed,
            up_norms,
            down_packed,
            down_norms,
            gate_codebook,
            down_codebook,
            gate_signs,
            down_signs,
        ):
            x_rot = hadamard_rotate_metal(x_flat, gate_signs)
            x_act = fused_gate_up(
                x_rot,
                gate_packed,
                gate_norms,
                up_packed,
                up_norms,
                gate_codebook,
                indices_flat,
            )
            x_act_rot = hadamard_rotate_metal(x_act, down_signs)
            return gather_down(
                x_act_rot,
                down_packed,
                down_norms,
                down_codebook,
                indices_flat,
            )

        fn = mx.compile(_mlp)
        tq_compiled[key] = fn
        return fn

    def _fast_switchglu_call(self, x, indices):
        gp = getattr(self, "gate_proj", None)
        up = getattr(self, "up_proj", None)
        dp = getattr(self, "down_proj", None)
        if (
            TurboQuantSwitchLinear is not None
            and isinstance(gp, TurboQuantSwitchLinear)
            and isinstance(up, TurboQuantSwitchLinear)
            and isinstance(dp, TurboQuantSwitchLinear)
        ):
            if getattr(self, "training", False):
                _record_fallback("tq_training_enabled")
                return original_call(self, x, indices)
            if getattr(indices, "ndim", 0) < 1 or int(getattr(indices, "size", 0) or 0) >= 64:
                _record_fallback("tq_prefill_or_large_indices")
                return original_call(self, x, indices)
            x_sq = x
            while getattr(x_sq, "ndim", 0) > 2 and x_sq.shape[-2] == 1:
                x_sq = x_sq.squeeze(-2)
            x_flat = x_sq.reshape(-1, gp.in_features)
            if int(x_flat.shape[0]) != 1:
                _record_fallback("tq_batched_or_prefill_x")
                return original_call(self, x, indices)
            k = int(indices.shape[-1]) if getattr(indices, "shape", ()) else 1
            if k <= 0:
                _record_fallback("tq_empty_indices")
                return original_call(self, x, indices)
            fn = _compiled_tq_for(gp, up, dp, k)
            try:
                out = fn(
                    x_flat.astype(mx.float32),
                    indices.reshape(-1).astype(mx.uint32),
                    gp.packed,
                    gp.norms,
                    up.packed,
                    up.norms,
                    dp.packed,
                    dp.norms,
                    gp.codebook,
                    dp.codebook,
                    gp.signs,
                    dp.signs,
                )
            except Exception:
                _record_fallback("tq_compiled_fast_path_error")
                logger.exception("MiMo-V2 TurboQuant SwitchGLU decode fast path failed; falling back")
                return original_call(self, x, indices)
            try:
                calls = int(getattr(SwitchGLU, "_vmlx_mimo_tq_decode_fast_calls", 0)) + 1
                SwitchGLU._vmlx_mimo_tq_decode_fast_calls = calls
                if calls in {1, 64, 512, 4096}:
                    logger.info(
                        "MiMo-V2 TurboQuant SwitchGLU decode fast path active: calls=%d compiled_shapes=%d",
                        calls,
                        len(tq_compiled),
                    )
            except Exception:
                pass
            if out.dtype != x.dtype:
                out = out.astype(x.dtype)
            return out.reshape(*indices.shape[:-1], k, gp.in_features)

        required_attrs = ("weight", "scales", "group_size", "bits", "mode")
        if not all(all(hasattr(module, attr) for attr in required_attrs) for module in (gp, up, dp)):
            _record_fallback("missing_affine_quant_attrs")
            return original_call(self, x, indices)
        if getattr(self, "training", False):
            _record_fallback("training_enabled")
            return original_call(self, x, indices)
        if getattr(indices, "ndim", 0) < 1 or int(getattr(indices, "size", 0) or 0) >= 64:
            _record_fallback("prefill_or_large_indices")
            return original_call(self, x, indices)

        x_sq = x
        while getattr(x_sq, "ndim", 0) > 2 and x_sq.shape[-2] == 1:
            x_sq = x_sq.squeeze(-2)
        x_flat = x_sq.reshape(-1, gp.input_dims)
        if int(x_flat.shape[0]) != 1:
            _record_fallback("batched_or_prefill_x")
            return original_call(self, x, indices)
        k = int(indices.shape[-1]) if getattr(indices, "shape", ()) else 1
        if k <= 0:
            _record_fallback("empty_indices")
            return original_call(self, x, indices)

        fn = _compiled_for(gp, up, dp, k)
        try:
            out = fn(
                x_flat.astype(mx.float32),
                indices.reshape(-1).astype(mx.uint32),
                gp["weight"],
                gp["scales"],
                gp.get("biases"),
                up["weight"],
                up["scales"],
                up.get("biases"),
                dp["weight"],
                dp["scales"],
                dp.get("biases"),
            )
        except Exception:
            _record_fallback("compiled_fast_path_error")
            logger.exception("MiMo-V2 affine SwitchGLU decode fast path failed; falling back")
            return original_call(self, x, indices)
        try:
            calls = int(getattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_calls", 0)) + 1
            SwitchGLU._vmlx_mimo_affine_decode_fast_calls = calls
            if calls in {1, 64, 512, 4096}:
                logger.info(
                    "MiMo-V2 affine SwitchGLU decode fast path active: calls=%d compiled_shapes=%d",
                    calls,
                    len(compiled),
                )
        except Exception:
            pass
        if out.dtype != x.dtype:
            out = out.astype(x.dtype)
        return out.reshape(*indices.shape[:-1], k, gp.input_dims)

    SwitchGLU._vmlx_mimo_original_affine_decode_call = original_call
    SwitchGLU._vmlx_mimo_affine_decode_fast_calls = 0
    SwitchGLU._vmlx_mimo_tq_decode_fast_calls = 0
    SwitchGLU._vmlx_mimo_affine_decode_fallback_reasons = {}
    SwitchGLU.__call__ = _fast_switchglu_call
    SwitchGLU._vmlx_mimo_affine_decode_fast_path = True
    logger.info("Installed vMLX MiMo-V2 affine SwitchGLU decode fast path")
    return True


def _is_mimo_v2_runtime_object_or_name(model: Any, model_name: Any = None) -> bool:
    """Return true when an MLLM-loaded object is the MiMo-V2 runtime.

    MiMo-V2 has an asymmetric native SWA cache contract. The generic
    TurboQuant-KV auto path flattens all layers into TurboQuantKVCache and
    breaks both speed and correctness for this architecture, so the MLLM load
    call site must skip it before invoking the generic tokenizer helper.
    """

    if "mimo" in str(model_name or "").lower():
        return True

    seen: set[int] = set()
    stack = [model]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if str(getattr(current, "model_type", "") or "").lower() == "mimo_v2":
            return True

        for attr in ("config", "args", "text_config"):
            value = getattr(current, attr, None)
            if isinstance(value, dict):
                if str(value.get("model_type", "") or "").lower() == "mimo_v2":
                    return True
                if str(value.get("architectures", "") or "").lower().find("mimo") >= 0:
                    return True
            elif value is not None:
                if str(getattr(value, "model_type", "") or "").lower() == "mimo_v2":
                    return True
                stack.append(value)

        for attr in ("inner", "language_model", "model"):
            value = getattr(current, attr, None)
            if value is not None:
                stack.append(value)

    return False


def _quantize_mimo_v2_runtime_modules(root: Any, filtered: dict, quantization: dict) -> int:
    """Quantize MiMo-V2 modules that live inside plain Python layer lists.

    `jang_tools.mimo_v2` stores decoder layers in `model.layers`, a normal
    Python list. MLX `nn.quantize(root, ...)` does not descend through that
    list from the text model root, so the attention projections and SwitchGLU
    expert projections stay unquantized unless vMLX walks each layer directly.
    """

    import mlx.nn as nn

    quantized_paths: set[str] = set()

    def params_for(path: str, module: Any):
        if not path or not hasattr(module, "to_quantized"):
            return False
        override = quantization.get(path)
        if isinstance(override, dict):
            quantized_paths.add(path)
            return dict(override)
        if f"{path}.scales" in filtered:
            quantized_paths.add(path)
            return True
        return False

    def quantize_module(module: Any, prefix: str = "") -> None:
        def class_predicate(path, child):
            full_path = f"{prefix}.{path}" if prefix and path else (prefix or path)
            return params_for(full_path, child)

        nn.quantize(
            module,
            group_size=quantization.get("group_size"),
            bits=quantization.get("bits"),
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    def _infer_affine_params(path: str, module: Any) -> dict[str, Any]:
        override = quantization.get(path)
        if isinstance(override, dict):
            return dict(override)
        group_size = int(quantization.get("group_size") or 64)
        bits = int(quantization.get("bits") or 8)
        weight = filtered.get(f"{path}.weight")
        scales = filtered.get(f"{path}.scales")
        module_weight = getattr(module, "weight", None)
        try:
            in_features = int(module_weight.shape[-1])
            scale_cols = int(scales.shape[-1])
            packed_cols = int(weight.shape[-1])
            if scale_cols > 0:
                inferred_group = in_features // scale_cols
                if inferred_group > 0:
                    group_size = inferred_group
                    inferred_bits = int(round(32 * packed_cols / max(in_features, 1)))
                    if inferred_bits in {2, 3, 4, 5, 6, 8}:
                        bits = inferred_bits
        except Exception:
            pass
        return {
            "group_size": group_size,
            "bits": bits,
            "mode": quantization.get("mode", "affine"),
        }

    def quantize_known_affine_path(owner: Any, attr: str, path: str) -> None:
        module = getattr(owner, attr, None)
        if module is None or not hasattr(module, "to_quantized"):
            return
        if f"{path}.weight" not in filtered or f"{path}.scales" not in filtered:
            return
        params = _infer_affine_params(path, module)
        setattr(owner, attr, module.to_quantized(**params))
        quantized_paths.add(path)

    backbone = getattr(root, "model", None)
    layers = getattr(backbone, "layers", None)
    if isinstance(layers, list):
        for idx, layer in enumerate(layers):
            quantize_module(layer, f"model.layers.{idx}")
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                # MiMo JANG_2L stores qkv_proj as MLX affine packed
                # uint32/scales/biases. The module lives under a normal Python
                # list in jang_tools, and some MLX quantize traversals miss it,
                # leaving a plain Linear with packed-width weights such as
                # [13568, 1024]. That crashes prefill against hidden size 4096.
                # Patch it from the artifact sidecars before load_weights so
                # the real affine QuantizedLinear receives the packed tensors.
                for qkv_path in (
                    f"model.layers.{idx}.self_attn.qkv_proj",
                    f"language_model.model.layers.{idx}.self_attn.qkv_proj",
                ):
                    before = len(quantized_paths)
                    quantize_known_affine_path(self_attn, "qkv_proj", qkv_path)
                    if len(quantized_paths) != before:
                        break

    # Quantize root-level leaves such as lm_head. Decoder layers are already
    # handled above because they are not visible from root.leaf_modules().
    quantize_module(root)
    return len(quantized_paths)


def _quantize_mimo_v2_passthrough_decode_hotspots(root: Any, quantization: dict) -> int:
    """Runtime-quantize MiMo decode-hot passthrough weights missing sidecars.

    The MiMo JANG contract requires `lm_head` and embeddings to be 8-bit affine,
    but some local bundles preserved them as bf16 without `.scales/.biases`.
    `lm_head` is a 152k-vocab projection paid every decode token, so leaving it
    bf16 makes the server appear to decode at ~1 tok/s even when the routed MoE
    path is quantized. This converts only known MiMo hotspots after weights are
    loaded, never generic model layers.
    """

    count = 0
    group_size = int(quantization.get("group_size") or 64)
    bits = int(quantization.get("bits") or 8)
    mode = str(quantization.get("mode") or "affine")

    lm_head = getattr(root, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "to_quantized"):
        root.lm_head = lm_head.to_quantized(group_size=group_size, bits=bits, mode=mode)
        count += 1

    backbone = getattr(root, "model", None)
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if embed_tokens is not None and hasattr(embed_tokens, "to_quantized"):
        backbone.embed_tokens = embed_tokens.to_quantized(
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        count += 1

    return count


def _upgrade_mimo_v2_loaded_qkv_affine_modules(root: Any, quantization: dict) -> int:
    """Upgrade loaded MiMo qkv packed-affine tensors inside Python layer lists.

    MiMo stores decoder layers in a normal Python list, so generic
    ``named_modules`` post-load repair does not reliably see qkv modules. If an
    affine JANG bundle loaded packed uint32 qkv tensors into a plain Linear, the
    first prefill crashes with hidden size 4096 vs packed width 1024. Walk the
    list directly and splice in QuantizedLinear modules from the loaded sidecar
    shapes.
    """

    import mlx.core as mx
    import mlx.nn as nn

    count = 0
    default_bits = int(quantization.get("bits") or 8)
    default_group_size = int(quantization.get("group_size") or 64)
    default_mode = str(quantization.get("mode") or "affine")

    backbone = getattr(root, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, list):
        return 0

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        module = getattr(self_attn, "qkv_proj", None)
        if self_attn is None or module is None:
            continue
        if isinstance(module, nn.QuantizedLinear):
            continue
        w = getattr(module, "weight", None)
        s = getattr(module, "scales", None)
        if w is None or s is None:
            continue
        if getattr(w, "dtype", None) != mx.uint32:
            continue

        try:
            packed_cols = int(w.shape[-1])
            scale_cols = int(s.shape[-1])
            inferred = None
            for bits in (8, 6, 4, 3, 2):
                real_cols = packed_cols * 32 // bits
                if scale_cols > 0 and real_cols % scale_cols == 0:
                    group_size = real_cols // scale_cols
                    if group_size in (32, 64, 128):
                        inferred = (bits, group_size, real_cols)
                        break
            if inferred is None:
                bits = default_bits
                group_size = default_group_size
                real_cols = packed_cols * 32 // bits
            else:
                bits, group_size, real_cols = inferred
            qmod = nn.QuantizedLinear(
                input_dims=real_cols,
                output_dims=int(w.shape[0]),
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=default_mode,
            )
            qmod.weight = w
            qmod.scales = s
            b = getattr(module, "biases", None)
            if b is not None:
                qmod.biases = b
            self_attn.qkv_proj = qmod
            count += 1
        except Exception:
            logger.exception("MiMo-V2 qkv packed affine upgrade failed")
    return count


def _install_mimo_v2_pending_qkv_affine_modules(
    root: Any,
    pending: dict[str, dict[str, Any]],
    quantization: dict,
) -> int:
    """Install qkv QuantizedLinear modules once split shard sidecars complete."""

    import re
    import mlx.nn as nn

    default_bits = int(quantization.get("bits") or 8)
    default_group_size = int(quantization.get("group_size") or 64)
    default_mode = str(quantization.get("mode") or "affine")
    backbone = getattr(root, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, list):
        return 0

    count = 0
    for base, parts in list(pending.items()):
        if "weight" not in parts or "scales" not in parts:
            continue
        match = re.match(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj$", base)
        if not match:
            continue
        layer_idx = int(match.group(1))
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        self_attn = getattr(layers[layer_idx], "self_attn", None)
        module = getattr(self_attn, "qkv_proj", None)
        if self_attn is None or module is None or isinstance(module, nn.QuantizedLinear):
            continue
        weight = parts["weight"]
        scales = parts["scales"]
        try:
            packed_cols = int(weight.shape[-1])
            scale_cols = int(scales.shape[-1])
            inferred = None
            for bits in (8, 6, 4, 3, 2):
                real_cols = packed_cols * 32 // bits
                if scale_cols > 0 and real_cols % scale_cols == 0:
                    group_size = real_cols // scale_cols
                    if group_size in (32, 64, 128):
                        inferred = (bits, group_size, real_cols)
                        break
            if inferred is None:
                bits = default_bits
                group_size = default_group_size
                real_cols = packed_cols * 32 // bits
            else:
                bits, group_size, real_cols = inferred
            qmod = nn.QuantizedLinear(
                input_dims=real_cols,
                output_dims=int(weight.shape[0]),
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=default_mode,
            )
            qmod.weight = weight
            qmod.scales = scales
            if parts.get("biases") is not None:
                qmod.biases = parts["biases"]
            self_attn.qkv_proj = qmod
            count += 1
        except Exception:
            logger.exception("MiMo-V2 pending qkv affine install failed for %s", base)
    return count


def _install_mimo_v2_pending_dense_mlp_affine_modules(
    root: Any,
    pending: dict[str, dict[str, Any]],
    quantization: dict,
) -> int:
    """Install dense MiMo MLP QuantizedLinear modules from split shard sidecars."""

    import re
    import mlx.nn as nn

    default_bits = int(quantization.get("bits") or 8)
    default_group_size = int(quantization.get("group_size") or 64)
    default_mode = str(quantization.get("mode") or "affine")
    backbone = getattr(root, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, list):
        return 0

    count = 0
    for base, parts in list(pending.items()):
        if "weight" not in parts or "scales" not in parts:
            continue
        match = re.match(
            r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$",
            base,
        )
        if not match:
            continue
        layer_idx = int(match.group(1))
        attr = match.group(2)
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        mlp = getattr(layers[layer_idx], "mlp", None)
        module = getattr(mlp, attr, None)
        if mlp is None or module is None or isinstance(module, nn.QuantizedLinear):
            continue
        weight = parts["weight"]
        scales = parts["scales"]
        try:
            packed_cols = int(weight.shape[-1])
            scale_cols = int(scales.shape[-1])
            inferred = None
            for bits in (8, 6, 4, 3, 2):
                real_cols = packed_cols * 32 // bits
                if scale_cols > 0 and real_cols % scale_cols == 0:
                    group_size = real_cols // scale_cols
                    if group_size in (32, 64, 128):
                        inferred = (bits, group_size, real_cols)
                        break
            if inferred is None:
                bits = default_bits
                group_size = default_group_size
                real_cols = packed_cols * 32 // bits
            else:
                bits, group_size, real_cols = inferred
            qmod = nn.QuantizedLinear(
                input_dims=real_cols,
                output_dims=int(weight.shape[0]),
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=default_mode,
            )
            qmod.weight = weight
            qmod.scales = scales
            if parts.get("biases") is not None:
                qmod.biases = parts["biases"]
            setattr(mlp, attr, qmod)
            count += 1
        except Exception:
            logger.exception(
                "MiMo-V2 pending dense MLP affine install failed for %s",
                base,
            )
    return count


def _install_mimo_v2_rotating_cache_keep_patch_if_needed(text_runtime: Any) -> bool:
    """Patch stale MiMo SWA cache construction to match full-forward windows.

    The bundled MiMo text runtime builds SWA layers with
    `RotatingKVCache(max_size=sliding_window, keep=4)`. That preserves the
    first four prompt tokens during incremental decode, but MiMo full-forward
    uses a plain sliding-window mask without a keep prefix. Live required-tool
    probes showed the mismatch directly:

    - full forward after `<parameter=value>blue` ranks `-cat` first;
    - incremental decode with `keep=4` ranks newline/punctuation first;
    - incremental decode with `keep=0` ranks `-cat` first.

    This patch changes only the MiMo cache constructor. It does not synthesize
    tool calls or alter model weights/logits outside the cache layout.
    """

    model_cls = getattr(text_runtime, "Model", None)
    rotating_cls = getattr(text_runtime, "RotatingKVCache", None)
    kv_cls = getattr(text_runtime, "KVCache", None)
    if (
        model_cls is None
        or rotating_cls is None
        or kv_cls is None
        or getattr(model_cls, "_vmlx_mimo_keep0_cache_patched", False)
    ):
        return False

    original_make_cache = model_cls.make_cache

    def _vmlx_mimo_keep0_make_cache(self):
        caches = []
        for layer in self.model.layers:
            if getattr(layer, "is_swa", False):
                caches.append(
                    rotating_cls(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(kv_cls())
        return caches

    model_cls._vmlx_original_make_cache = original_make_cache
    model_cls.make_cache = _vmlx_mimo_keep0_make_cache
    model_cls._vmlx_mimo_keep0_cache_patched = True
    logger.info("Installed vMLX MiMo-V2 keep=0 rotating-cache patch")
    return True


def _register_mimo_v2_mlx_vlm_runtime() -> None:
    import mlx.core as mx

    module_name = "mlx_vlm.models.mimo_v2"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None and getattr(
        existing_module, "_vmlx_mimo_v2_runtime_registered", False
    ):
        return

    text_runtime = importlib.import_module("jang_tools.mimo_v2.mlx_model")
    _install_mimo_v2_compiled_router_if_missing(text_runtime)
    _install_mimo_v2_affine_switchglu_decode_fast_path()
    _install_mimo_v2_rotating_cache_keep_patch_if_needed(text_runtime)

    import mlx.nn as nn

    TextConfig = getattr(text_runtime, "ModelArgs")
    TextModel = getattr(text_runtime, "Model")

    _INPUTS_EMBEDS_UNSET = object()

    def _mimo_v2_get_module(root, dotted):
        cur = root
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        return cur

    def _mimo_v2_set_module(root, dotted, new_module):
        parts = dotted.split(".")
        cur = root
        for part in parts[:-1]:
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = new_module
        else:
            setattr(cur, last, new_module)

    def _mimo_v2_assign_weight(root, dotted, value):
        if dotted == "visual.patch_embed.proj.weight" and len(value.shape) > 2:
            value = mx.reshape(value, (value.shape[0], -1))
        if (
            dotted.startswith("visual.blocks.")
            and dotted.endswith(".attn.qkv.weight")
            and len(value.shape) == 2
        ):
            attn_path = dotted[: -len(".qkv.weight")]
            attn = _mimo_v2_get_module(root, attn_path)
            denom = int(attn.num_heads) + 2 * int(attn.num_kv_heads)
            if denom > 0 and int(value.shape[0]) % denom == 0:
                inferred_head_dim = int(value.shape[0]) // denom
                attn.head_dim = inferred_head_dim
                attn.scaling = inferred_head_dim**-0.5
                try:
                    visual = _mimo_v2_get_module(root, "visual")
                    visual.vision_head_dim = inferred_head_dim
                except Exception:
                    pass
        parts = dotted.split(".")
        cur = root
        for part in parts[:-1]:
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = value
        else:
            setattr(cur, last, value)

    def _install_mimo_v2_prestacked_jangtq_modules(model, weights, *, seed=42):
        """Bind prestacked MiMo JANGTQ tensors to TurboQuant switch modules.

        The promoted MiMo-V2.5-JANGTQ_2 bundle stores routed experts directly as
        ``model.layers.N.mlp.switch_mlp.{proj}.tq_{packed,norms,bits}``.
        ``mlx_vlm.load`` calls this shim's ``load_weights`` path, which otherwise
        rejects those keys as unknown parameters. Do not silence strict loading:
        replace the corresponding SwitchLinear with TurboQuantSwitchLinear and
        assign the actual packed/norm tensors from the artifact.
        """
        tq_groups = {}
        for key, value in weights.items():
            if key.endswith(".tq_packed"):
                tq_groups.setdefault(key[: -len(".tq_packed")], {})["packed"] = value
            elif key.endswith(".tq_norms"):
                tq_groups.setdefault(key[: -len(".tq_norms")], {})["norms"] = value
            elif key.endswith(".tq_bits"):
                tq_groups.setdefault(key[: -len(".tq_bits")], {})["bits"] = value

        if not tq_groups:
            return 0

        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

        installed = 0
        for base, parts in sorted(tq_groups.items()):
            if ".switch_mlp." not in base:
                continue
            missing = {"packed", "norms", "bits"} - set(parts)
            if missing:
                raise RuntimeError(
                    f"MiMo-V2 prestacked JANGTQ group {base} missing {sorted(missing)}"
                )

            packed = parts["packed"]
            norms = parts["norms"]
            bits_tensor = parts["bits"]
            try:
                bits = int(bits_tensor[0].item())
            except Exception:
                bits = int(bits_tensor.item())
            if bits <= 0 or 32 % bits != 0:
                raise RuntimeError(f"MiMo-V2 prestacked JANGTQ group {base} has bad bits={bits}")
            if len(packed.shape) != 3 or len(norms.shape) != 2:
                raise RuntimeError(
                    f"MiMo-V2 prestacked JANGTQ group {base} has bad shapes: "
                    f"packed={packed.shape}, norms={norms.shape}"
                )
            if packed.shape[:2] != norms.shape:
                raise RuntimeError(
                    f"MiMo-V2 prestacked JANGTQ group {base} shape mismatch: "
                    f"packed={packed.shape}, norms={norms.shape}"
                )

            # Fail loudly if the artifact names a module that the MiMo runtime
            # does not expose. This distinguishes runtime incompatibility from
            # corruption instead of silently dropping expert weights.
            _mimo_v2_get_module(model, base)

            num_experts, out_features, packed_cols = packed.shape
            in_features = packed_cols * (32 // bits)
            tq_module = TurboQuantSwitchLinear(
                in_features=in_features,
                out_features=out_features,
                num_experts=num_experts,
                bits=bits,
                bias=False,
                seed=seed,
            )
            tq_module.packed = packed
            tq_module.norms = norms
            _mimo_v2_set_module(model, base, tq_module)
            installed += 1

        return installed

    class _MiMoV2CausalLMOutput:
        def __init__(self, logits):
            self.logits = logits
            self.cross_attention_states = None
            self.encoder_outputs = None

    class _MiMoV2LanguageModelCompat(nn.Module):
        """Adapt jang_tools' mlx-lm MiMo model to mlx-vlm's language-model API."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @property
        def model(self):
            return self.inner.model

        @property
        def layers(self):
            return self.inner.layers

        def make_cache(self):
            return self.inner.make_cache()

        def sanitize(self, weights):
            return self.inner.sanitize(weights)

        def load_weights(self, weights, strict=True):
            return self.inner.load_weights(weights, strict=strict)

        def __call__(
            self,
            input_ids=None,
            *,
            inputs_embeds=_INPUTS_EMBEDS_UNSET,
            cache=None,
            mask=None,
            **kwargs,
        ):
            mllm_style_call = inputs_embeds is not _INPUTS_EMBEDS_UNSET
            if mllm_style_call:
                try:
                    logits = self.inner(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        cache=cache,
                        mask=mask,
                        **kwargs,
                    )
                except TypeError as exc:
                    if "inputs_embeds" not in str(exc):
                        raise
                    if inputs_embeds is None:
                        logits = self.inner(input_ids, cache=cache, mask=mask, **kwargs)
                    else:
                        logits = self.inner.model(
                            input_ids=None,
                            inputs_embeds=inputs_embeds,
                            cache=cache,
                            mask=mask,
                            **kwargs,
                        )
                        logits = self.inner.lm_head(logits)
                if hasattr(logits, "logits"):
                    return logits
                return _MiMoV2CausalLMOutput(logits)
            return self.inner(input_ids, cache=cache, mask=mask, **kwargs)

    class _MiMoV2MediaConfig:
        """Preserve MiMo media metadata until the real media forward is wired.

        The earlier placeholder configs threw away `vision_config`,
        `audio_config`, and `processor_config` even though the local JANG
        bundles carry the exact ViT/audio/tokenizer contract. Keeping these
        fields on the runtime config is a required intermediate step for real
        VL/audio/video wiring: load/forward code can now consume the model-owned
        shapes, token ids, pixel/frame limits, and timing parameters instead of
        guessing or hard-coding them.
        """

        default_model_type = "mimo_v2_media"

        def __init__(self, **params):
            self.raw_config = dict(params)
            self.model_type = str(
                params.get("model_type") or self.default_model_type
            )
            for key, value in params.items():
                setattr(self, key, value)

        @classmethod
        def from_dict(cls, params):
            return cls(**dict(params or {}))

        def to_dict(self):
            return dict(self.raw_config)

    class VisionConfig(_MiMoV2MediaConfig):
        default_model_type = "mimo_v2_vision"

    class AudioConfig(_MiMoV2MediaConfig):
        default_model_type = "mimo_v2_audio"

    class ModelConfig:
        def __init__(
            self,
            text_config=None,
            vision_config=None,
            audio_config=None,
            processor_config=None,
            model_type: str = "mimo_v2",
            eos_token_id=None,
            **kwargs,
        ):
            self.model_type = model_type
            self.text_config = text_config
            self.vision_config = vision_config or VisionConfig()
            self.audio_config = audio_config or AudioConfig()
            self.processor_config = dict(processor_config or {})
            self.eos_token_id = eos_token_id
            self.image_token_id = kwargs.get("image_token_id")
            self.video_token_id = kwargs.get("video_token_id")
            self.audio_token_id = kwargs.get("audio_token_id")
            self.vision_start_token_id = kwargs.get("vision_start_token_id")
            self.vision_end_token_id = kwargs.get("vision_end_token_id")
            self.video_start_token_id = kwargs.get("video_start_token_id")
            self.video_end_token_id = kwargs.get("video_end_token_id")
            self.audio_start_token_id = kwargs.get("audio_start_token_id")
            self.audio_end_token_id = kwargs.get("audio_end_token_id")
            self.media_token_ids = {
                key: value
                for key, value in {
                    "image": self.image_token_id,
                    "video": self.video_token_id,
                    "audio": self.audio_token_id,
                    "vision_start": self.vision_start_token_id,
                    "vision_end": self.vision_end_token_id,
                    "video_start": self.video_start_token_id,
                    "video_end": self.video_end_token_id,
                    "audio_start": self.audio_start_token_id,
                    "audio_end": self.audio_end_token_id,
                }.items()
                if value is not None
            }
            self.media_runtime_status = kwargs.get(
                "multimodal_status",
                "weights_preserved_text_runtime",
            )
            self.preserved_modalities = list(
                kwargs.get("preserved_modalities") or ["vision", "image", "video", "audio"]
            )
            self.unwired_modalities = list(
                kwargs.get("unwired_modalities") or ["vision", "image", "video", "audio"]
            )

        @property
        def media_runtime_wired(self) -> bool:
            status = str(self.media_runtime_status or "").lower()
            if status in {
                "",
                "weights_preserved_text_runtime",
                "text_runtime",
                "text_only",
                "unwired",
                "preserved_disabled",
            }:
                return False
            return not bool(self.unwired_modalities)

        @classmethod
        def from_dict(cls, params):
            merged_text = dict(params)
            if isinstance(params.get("text_config"), dict):
                merged_text.update(params["text_config"])
            params["text_config"] = merged_text
            quantization = params.get("quantization")
            if isinstance(quantization, dict):
                for key, value in list(quantization.items()):
                    if (
                        isinstance(value, dict)
                        and key.startswith(("model.", "lm_head"))
                        and not key.startswith("language_model.")
                    ):
                        quantization.setdefault(f"language_model.{key}", value)
            return cls(
                text_config=TextConfig.from_dict(merged_text),
                vision_config=VisionConfig.from_dict(params.get("vision_config") or {}),
                audio_config=AudioConfig.from_dict(params.get("audio_config") or {}),
                processor_config=params.get("processor_config") or {},
                model_type=str(params.get("model_type") or "mimo_v2"),
                eos_token_id=params.get("eos_token_id"),
                image_token_id=params.get("image_token_id")
                or (params.get("processor_config") or {}).get("image_token_id"),
                video_token_id=params.get("video_token_id")
                or (params.get("processor_config") or {}).get("video_token_id"),
                audio_token_id=params.get("audio_token_id")
                or (params.get("processor_config") or {}).get("audio_token_id"),
                vision_start_token_id=params.get("vision_start_token_id")
                or (params.get("processor_config") or {}).get("vision_start_token_id"),
                vision_end_token_id=params.get("vision_end_token_id")
                or (params.get("processor_config") or {}).get("vision_end_token_id"),
                video_start_token_id=(params.get("processor_config") or {}).get("video_start_token_id"),
                video_end_token_id=(params.get("processor_config") or {}).get("video_end_token_id"),
                audio_start_token_id=(params.get("processor_config") or {}).get("audio_start_token_id"),
                audio_end_token_id=(params.get("processor_config") or {}).get("audio_end_token_id"),
                preserved_modalities=(
                    (params.get("capabilities") or {}).get("preserved_modalities")
                    or params.get("preserved_modalities")
                ),
                unwired_modalities=(
                    (params.get("capabilities") or {}).get("unwired_modalities")
                    or params.get("unwired_modalities")
                ),
                multimodal_status=(
                    (params.get("capabilities") or {}).get("multimodal_status")
                    or params.get("multimodal_status")
                ),
            )

    class VisionModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config or VisionConfig()
            self.hidden_size = int(getattr(self.config, "hidden_size", 1280))
            self.out_hidden_size = int(
                getattr(self.config, "out_hidden_size", 4096)
            )
            self.patch_size = int(getattr(self.config, "patch_size", 16))
            self.temporal_patch_size = int(
                getattr(self.config, "temporal_patch_size", 2)
            )
            self.in_channels = int(
                getattr(self.config, "in_channels", None)
                or getattr(self.config, "in_chans", 3)
            )
            self.spatial_merge_size = int(
                getattr(self.config, "spatial_merge_size", 2)
            )
            self.patch_embed = MiMoVisionPatchEmbed(
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                in_channels=self.in_channels,
                embed_dim=self.hidden_size,
            )
            self.merger = MiMoVisionPatchMerger(
                dim=self.out_hidden_size,
                context_dim=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
            )
            depth = int(getattr(self.config, "depth", 0) or 0)
            intermediate_size = int(
                getattr(self.config, "intermediate_size", self.hidden_size * 4)
            )
            num_heads = int(getattr(self.config, "num_heads", 1) or 1)
            num_kv_heads = int(
                getattr(self.config, "num_key_value_heads", num_heads) or num_heads
            )
            head_dim = int(getattr(self.config, "qk_channels", 0) or 0)
            if head_dim <= 0:
                head_dim = max(1, self.hidden_size // max(1, num_heads))
            self.vision_head_dim = head_dim
            rms_norm_eps = float(getattr(self.config, "rms_norm_eps", 1e-6) or 1e-6)
            hidden_act = str(getattr(self.config, "hidden_act", "silu") or "silu")
            use_sink = bool(getattr(self.config, "use_sink", False))
            window_size = int(getattr(self.config, "visual_token_window_size", -1) or -1)
            self.fullatt_block_indexes = set(
                int(i) for i in (getattr(self.config, "fullatt_block_indexes", []) or [])
            )
            self.vit_window_attn_types = list(
                getattr(self.config, "vit_window_attn_types", None) or [-1] * depth
            )
            self.blocks = [
                MiMoVisionBlock(
                    dim=self.hidden_size,
                    intermediate_dim=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_act=hidden_act,
                    rms_norm_eps=rms_norm_eps,
                    use_sinks=use_sink and (i not in self.fullatt_block_indexes),
                    window_size=window_size,
                )
                for i in range(depth)
            ]

        def embed_patches(self, pixel_values):
            return self.patch_embed(pixel_values)

        def merge_patches(self, hidden_states):
            return self.merger(hidden_states)

        def _grid_rows(self, grid_thw):
            grid = mx.array(grid_thw)
            if grid.ndim == 1:
                grid = grid[None, :]
            return [[int(v) for v in row] for row in grid.tolist()]

        def apply_index(self, tensor, index):
            tensor = mx.array(tensor)
            index = mx.array(index)
            unit = self.spatial_merge_size * self.spatial_merge_size
            tail_shape = tuple(tensor.shape[1:])
            grouped = mx.reshape(tensor, (-1, unit) + tail_shape)
            selected = grouped[index]
            return mx.reshape(selected, (-1,) + tail_shape)

        def get_window_index_1d(self, grid_thw, col: bool = True):
            window_index = []
            window_index_id = 0
            for grid_t, grid_h, grid_w in self._grid_rows(grid_thw):
                llm_grid_h = grid_h // self.spatial_merge_size
                llm_grid_w = grid_w // self.spatial_merge_size
                index = mx.reshape(
                    mx.arange(grid_t * llm_grid_h * llm_grid_w),
                    (grid_t, llm_grid_h, llm_grid_w),
                )
                if col:
                    index = mx.reshape(mx.transpose(index, (0, 2, 1)), (-1,))
                else:
                    index = mx.reshape(index, (-1,))
                window_index.extend((index + window_index_id).tolist())
                window_index_id += grid_t * llm_grid_h * llm_grid_w
            return mx.array(window_index, dtype=mx.int32)

        def rot_pos_emb(self, grid_thw):
            pos_ids = []
            max_grid_size = 1
            for grid_t, grid_h, grid_w in self._grid_rows(grid_thw):
                max_grid_size = max(max_grid_size, grid_h, grid_w)
                hpos = mx.broadcast_to(mx.arange(grid_h)[:, None], (grid_h, grid_w))
                hpos = mx.reshape(
                    hpos,
                    (
                        grid_h // self.spatial_merge_size,
                        self.spatial_merge_size,
                        grid_w // self.spatial_merge_size,
                        self.spatial_merge_size,
                    ),
                )
                hpos = mx.reshape(mx.transpose(hpos, (0, 2, 1, 3)), (-1,))
                wpos = mx.broadcast_to(mx.arange(grid_w)[None, :], (grid_h, grid_w))
                wpos = mx.reshape(
                    wpos,
                    (
                        grid_h // self.spatial_merge_size,
                        self.spatial_merge_size,
                        grid_w // self.spatial_merge_size,
                        self.spatial_merge_size,
                    ),
                )
                wpos = mx.reshape(mx.transpose(wpos, (0, 2, 1, 3)), (-1,))
                pos = mx.stack([hpos, wpos], axis=-1)
                if grid_t > 1:
                    pos = mx.tile(pos, (grid_t, 1))
                pos_ids.append(pos)
            pos_ids = mx.concatenate(pos_ids, axis=0)

            rotary_dim = max(2, self.vision_head_dim // 2)
            if rotary_dim % 2:
                rotary_dim += 1
            inv_freq = 1.0 / (
                10000.0 ** (mx.arange(0, rotary_dim, 2).astype(mx.float32) / rotary_dim)
            )
            seq = mx.arange(max_grid_size).astype(mx.float32)
            rotary = seq[:, None] * inv_freq[None, :]
            return mx.reshape(rotary[pos_ids], (pos_ids.shape[0], -1))

        def run_blocks(self, hidden_states, grid_thw=None):
            if grid_thw is None:
                row_based_embeddings = None
                col_based_embeddings = None
                window_index_1d_col = None
                reverse_window_index_1d_col = None
            else:
                emb = self.rot_pos_emb(grid_thw)
                emb = mx.concatenate([emb, emb], axis=-1)
                row_based_embeddings = (mx.cos(emb), mx.sin(emb))
                window_index_1d_col = self.get_window_index_1d(grid_thw, col=True)
                reverse_window_index_1d_col = mx.argsort(window_index_1d_col)
                col_emb = self.apply_index(emb, window_index_1d_col)
                col_based_embeddings = (mx.cos(col_emb), mx.sin(col_emb))

            for i, block in enumerate(self.blocks):
                window_attn_type = (
                    self.vit_window_attn_types[i]
                    if i < len(self.vit_window_attn_types)
                    else -1
                )
                if (
                    grid_thw is not None
                    and window_attn_type == 1
                    and (i == 0 or self.vit_window_attn_types[i - 1] != 1)
                ):
                    hidden_states = self.apply_index(hidden_states, window_index_1d_col)
                if (
                    grid_thw is not None
                    and i > 0
                    and window_attn_type != 1
                    and self.vit_window_attn_types[i - 1] == 1
                ):
                    hidden_states = self.apply_index(
                        hidden_states,
                        reverse_window_index_1d_col,
                    )
                full_attn = i in self.fullatt_block_indexes
                position_embeddings = (
                    col_based_embeddings
                    if grid_thw is not None and window_attn_type == 1
                    else row_based_embeddings
                )
                hidden_states = block(
                    hidden_states,
                    full_attn=full_attn,
                    position_embeddings=position_embeddings,
                )
            return hidden_states

        def __call__(self, pixel_values=None, grid_thw=None, **kwargs):
            if pixel_values is None or grid_thw is None:
                raise UnsupportedMediaModalityError(
                    "vision",
                    "MiMo-V2.5 vision forward requires pixel_values and grid_thw.",
                    family="mimo_v2",
                )
            x = self.embed_patches(pixel_values)
            x = self.run_blocks(x, grid_thw=grid_thw)
            return self.merge_patches(x)

        def sanitize(self, weights):
            return {
                key: value
                for key, value in weights.items()
                if not (
                    key.startswith("visual.")
                    or key.startswith("vision_tower.")
                    or key.startswith("audio_encoder.")
                    or key.startswith("speech_embeddings.")
                )
            }

    class AudioProjection(nn.Module):
        def __init__(self, *, input_size: int, hidden_size: int, output_size: int):
            super().__init__()
            self.mlp = [
                nn.Linear(input_size, hidden_size, bias=False),
                MiMoVisionGELU(),
                nn.Linear(hidden_size, output_size, bias=False),
            ]

        def __call__(self, x):
            for layer in self.mlp:
                x = layer(x)
            return x

    class MiMoAudioEuclideanCodebook(nn.Module):
        """MiMo audio tokenizer nearest-codebook encoder.

        Mirrors the official PyTorch `EuclideanCodebook.encode` path used by
        `MiMoAudioTokenizer`: maximize `-(||x||^2 - 2*x@embed.T + ||embed||^2)`.
        """

        def __init__(self, embed):
            super().__init__()
            self.embed = mx.array(embed)

        def encode(self, x):
            x = mx.array(x)
            original_shape = tuple(x.shape[:-1])
            flat = mx.reshape(x, (-1, x.shape[-1])).astype(mx.float32)
            embed = self.embed.astype(mx.float32)
            dist = -(
                mx.sum(flat * flat, axis=1, keepdims=True)
                - 2 * (flat @ mx.transpose(embed))
                + mx.sum(embed * embed, axis=1, keepdims=True).T
            )
            indices = mx.argmax(dist, axis=-1).astype(mx.int32)
            return mx.reshape(indices, original_shape)

        def decode(self, indices):
            return self.embed[mx.array(indices).astype(mx.int32)]

    class MiMoAudioResidualVectorQuantizer(nn.Module):
        """Residual VQ used by the MiMo audio tokenizer encoder output."""

        def __init__(self, codebook_embeds):
            super().__init__()
            self.codebooks = [MiMoAudioEuclideanCodebook(embed) for embed in codebook_embeds]

        def encode(self, hidden_states, n_q: int | None = None, st: int = 0):
            residual = mx.array(hidden_states)
            n_q = len(self.codebooks) if n_q is None else int(n_q)
            st = int(st or 0)
            if st < 0 or n_q < st or n_q > len(self.codebooks):
                raise ValueError(
                    f"Invalid MiMo audio RVQ range: st={st}, n_q={n_q}, "
                    f"codebooks={len(self.codebooks)}"
                )
            all_indices = []
            for codebook in self.codebooks[st:n_q]:
                indices = codebook.encode(residual)
                quantized = codebook.decode(indices).astype(residual.dtype)
                residual = residual - quantized
                all_indices.append(indices)
            if not all_indices:
                return mx.zeros((0,) + tuple(residual.shape[:-1]), dtype=mx.int32)
            return mx.stack(all_indices, axis=0)

        def decode(self, codes, st: int = 0):
            codes = mx.array(codes).astype(mx.int32)
            if codes.ndim < 1:
                raise ValueError("MiMo audio RVQ codes must include a quantizer axis")
            st = int(st or 0)
            if st < 0 or st + int(codes.shape[0]) > len(self.codebooks):
                raise ValueError(
                    f"Invalid MiMo audio RVQ decode range: st={st}, "
                    f"codes={codes.shape[0]}, codebooks={len(self.codebooks)}"
                )
            out = None
            for offset in range(int(codes.shape[0])):
                decoded = self.codebooks[st + offset].decode(codes[offset])
                out = decoded if out is None else out + decoded
            return out

    @dataclass
    class MiMoAudioTokenizerConfig:
        """MiMo audio-tokenizer config fields needed by the MLX runtime."""

        max_audio_seconds: int = 300
        stride_size: int = 2
        avg_pooler: int = 2
        d_model: int = 1024
        scale_embedding: bool = False
        kernel_size: int = 3
        activation_function: str = "gelu"
        encoder_layers: int = 24
        encoder_skip_layer_id: int | None = 3
        encoder_attention_heads: int = 16
        encoder_ffn_dim: int = 4096
        encoder_causal: bool = True
        encoder_attn_window_size: list[int] = field(default_factory=lambda: [128, 0])
        nfft: int = 960
        n_mels: int = 128
        sampling_rate: int = 24000
        hop_length: int = 240
        window_size: int = 960
        num_quantizers: int = 20
        codebook_size: list[int] = field(default_factory=list)
        threshold_ema_dead_code: int = 2
        position_embedding_type: str = "rope"
        rope_theta: int = 10000
        rope_type: str = "default"
        ln_type: str = "LayerNorm"
        hybrid_attention: bool = True
        hybrid_block_size: int = 8
        swa_per_block: int = 2

        @classmethod
        def from_dict(cls, data: dict[str, Any]):
            params = {}
            for field_name in cls.__dataclass_fields__:
                if field_name in data:
                    params[field_name] = data[field_name]
            config = cls(**params)
            if not config.codebook_size:
                config.codebook_size = [1024] * int(config.num_quantizers)
            return config

        @classmethod
        def from_bundle(cls, model_path: str | Path):
            root = Path(model_path)
            audio_dir = root if root.name == "audio_tokenizer" else root / "audio_tokenizer"
            config_path = audio_dir / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"MiMo audio tokenizer config missing: {config_path}"
                )
            return cls.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

        def get_output_length(self, mel_len):
            """Official Conv1d/stride output length before avg-pool downsample."""
            mel = mx.array(mel_len).astype(mx.int32)
            tgt_len = mel + 3 - int(self.kernel_size)
            return ((tgt_len + 2 - int(self.kernel_size)) // int(self.stride_size)) + 1

        def get_code_length(self, mel_len):
            """Official tokenizer code length after optional avg-pool downsample."""
            out = self.get_output_length(mel_len)
            avg = int(self.avg_pooler or 1)
            if avg == 1:
                return out
            return (out // avg) + ((out % avg) != 0).astype(mx.int32)

    def plan_mimo_audio_mel_segments(
        mels: list[Any],
        *,
        config: MiMoAudioTokenizerConfig | None = None,
        segment_size: int = 6000,
    ) -> dict[str, Any]:
        """Build the official MiMo audio-tokenizer mel segmentation plan.

        Mirrors `tokenize_audio_batch`: preserve the concatenated mel tensor,
        record per-segment input lengths, and precompute per-input code lengths.
        The encoder forward consumes this plan; this helper does not synthesize
        codes and does not replace the missing 24-layer encoder.
        """

        if not mels:
            return {
                "input_features": mx.zeros((0, int((config or MiMoAudioTokenizerConfig()).n_mels))),
                "input_lens": mx.zeros((0,), dtype=mx.int32),
                "segments_per_mel": [],
                "code_lengths": [],
            }
        config = config or MiMoAudioTokenizerConfig()
        segment_size = int(segment_size)
        if segment_size <= 0:
            raise ValueError("MiMo audio segment_size must be positive")
        arrays = [mx.array(mel) for mel in mels]
        for idx, arr in enumerate(arrays):
            if arr.ndim != 2:
                raise ValueError(
                    f"MiMo audio mel {idx} must be 2D [T, n_mels], got {arr.shape}"
                )
            if int(arr.shape[-1]) != int(config.n_mels):
                raise ValueError(
                    f"MiMo audio mel {idx} has n_mels={arr.shape[-1]}, "
                    f"expected {config.n_mels}"
                )
        segments_per_mel: list[list[int]] = []
        for arr in arrays:
            length = int(arr.shape[0])
            segments = [segment_size] * (length // segment_size)
            if length % segment_size:
                segments.append(length % segment_size)
            if not segments:
                segments.append(0)
            segments_per_mel.append(segments)
        input_lens = [length for segments in segments_per_mel for length in segments]
        input_features = mx.concatenate(arrays, axis=0)
        code_lengths = []
        for segments in segments_per_mel:
            seg_lengths = mx.array(segments, dtype=mx.int32)
            code_lengths.append(int(mx.sum(config.get_code_length(seg_lengths)).item()))
        return {
            "input_features": input_features,
            "input_lens": mx.array(input_lens, dtype=mx.int32),
            "segments_per_mel": segments_per_mel,
            "code_lengths": code_lengths,
        }

    def _mimo_audio_lengths_list(lengths) -> list[int]:
        arr = mx.array(lengths).astype(mx.int32)
        if arr.ndim == 0:
            return [int(arr.item())]
        return [int(x) for x in arr.tolist()]

    def _mimo_audio_unpack_segments(features, lengths):
        features = mx.array(features)
        lens = _mimo_audio_lengths_list(lengths)
        if not lens:
            return mx.zeros((0, 0, int(features.shape[-1]) if features.ndim else 0))
        max_len = max(lens)
        rows = []
        offset = 0
        for length in lens:
            row = features[offset : offset + length]
            offset += length
            if length < max_len:
                pad = mx.zeros((max_len - length, features.shape[-1]), dtype=features.dtype)
                row = mx.concatenate([row, pad], axis=0)
            rows.append(row[None, :, :])
        return mx.concatenate(rows, axis=0)

    def _mimo_audio_pack_segments(hidden, lengths):
        lens = _mimo_audio_lengths_list(lengths)
        rows = []
        for idx, length in enumerate(lens):
            if length > 0:
                rows.append(hidden[idx, :length, :])
        if not rows:
            return mx.zeros((0, hidden.shape[-1]), dtype=hidden.dtype)
        return mx.concatenate(rows, axis=0)

    def _mimo_audio_position_ids(lengths):
        lens = _mimo_audio_lengths_list(lengths)
        parts = [mx.arange(length, dtype=mx.float32) for length in lens if length > 0]
        if not parts:
            return mx.zeros((0,), dtype=mx.float32)
        return mx.concatenate(parts, axis=0)

    def _mimo_audio_rotate_half(x):
        half = int(x.shape[-1]) // 2
        first = x[..., :half]
        second = x[..., half : half * 2]
        rest = x[..., half * 2 :]
        rotated = mx.concatenate([-second, first], axis=-1)
        return mx.concatenate([rotated, rest], axis=-1) if rest.shape[-1] else rotated

    def _mimo_audio_activation(name: str, x):
        lowered = str(name or "gelu").lower()
        if lowered == "silu":
            return nn.silu(x)
        return nn.gelu(x)

    class MiMoAudioTokenizerAttention(nn.Module):
        def __init__(self, config: MiMoAudioTokenizerConfig, window_size=None):
            super().__init__()
            self.embed_dim = int(config.d_model)
            self.num_heads = int(config.encoder_attention_heads)
            self.head_dim = self.embed_dim // max(1, self.num_heads)
            self.scaling = self.head_dim**-0.5
            self.window_size = tuple(window_size or [-1, -1])
            self.causal = bool(config.encoder_causal)
            self.rope_theta = float(config.rope_theta)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        def _apply_rope(self, q, k, lengths):
            positions = _mimo_audio_position_ids(lengths)
            if int(positions.shape[0]) == 0:
                return q, k
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    mx.arange(0, self.head_dim, 2).astype(mx.float32)
                    / self.head_dim
                )
            )
            freqs = positions[:, None] * inv_freq[None, :]
            emb = mx.concatenate([freqs, freqs], axis=-1)
            cos = emb[:, None, :].astype(q.dtype)
            sin = emb[:, None, :].astype(q.dtype)
            return (
                q * cos + _mimo_audio_rotate_half(q) * sin,
                k * cos + _mimo_audio_rotate_half(k) * sin,
            )

        def _mask(self, seq_len: int, dtype):
            has_window = self.window_size and int(self.window_size[0]) > 0
            if not self.causal and not has_window:
                return None
            row = mx.arange(seq_len)[:, None]
            col = mx.arange(seq_len)[None, :]
            mask = mx.zeros((seq_len, seq_len), dtype=dtype)
            if self.causal:
                mask = mx.where(col > row, mx.full((seq_len, seq_len), -mx.inf), mask)
            if has_window:
                mask = mx.where(
                    mx.abs(row - col) > int(self.window_size[0]),
                    mx.full((seq_len, seq_len), -mx.inf),
                    mask,
                )
            return mask[None, None, :, :]

        def __call__(self, hidden_states, lengths):
            hidden_states = mx.array(hidden_states)
            total_len = int(hidden_states.shape[0])
            q = mx.reshape(self.q_proj(hidden_states), (total_len, self.num_heads, self.head_dim))
            k = mx.reshape(self.k_proj(hidden_states), (total_len, self.num_heads, self.head_dim))
            v = mx.reshape(self.v_proj(hidden_states), (total_len, self.num_heads, self.head_dim))
            q, k = self._apply_rope(q, k, lengths)
            outputs = []
            offset = 0
            for seq_len in _mimo_audio_lengths_list(lengths):
                if seq_len <= 0:
                    continue
                q_seq = mx.transpose(q[offset : offset + seq_len], (1, 0, 2))[None, :, :, :]
                k_seq = mx.transpose(k[offset : offset + seq_len], (1, 0, 2))[None, :, :, :]
                v_seq = mx.transpose(v[offset : offset + seq_len], (1, 0, 2))[None, :, :, :]
                attn = mx.fast.scaled_dot_product_attention(
                    q_seq,
                    k_seq,
                    v_seq,
                    scale=self.scaling,
                    mask=self._mask(seq_len, q_seq.dtype),
                )
                outputs.append(mx.reshape(mx.transpose(attn[0], (1, 0, 2)), (seq_len, self.embed_dim)))
                offset += seq_len
            if not outputs:
                return mx.zeros_like(hidden_states)
            return self.out_proj(mx.concatenate(outputs, axis=0))

    class MiMoAudioTokenizerLayer(nn.Module):
        def __init__(self, config: MiMoAudioTokenizerConfig, window_size=None):
            super().__init__()
            self.self_attn = MiMoAudioTokenizerAttention(config, window_size=window_size)
            self.self_attn_layer_norm = nn.LayerNorm(int(config.d_model))
            self.fc1 = nn.Linear(int(config.d_model), int(config.encoder_ffn_dim), bias=True)
            self.fc2 = nn.Linear(int(config.encoder_ffn_dim), int(config.d_model), bias=True)
            self.final_layer_norm = nn.LayerNorm(int(config.d_model))
            self.activation_function = str(config.activation_function or "gelu")

        def __call__(self, hidden_states, lengths):
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            hidden_states = self.self_attn(hidden_states, lengths)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = _mimo_audio_activation(
                self.activation_function,
                self.fc1(hidden_states),
            )
            hidden_states = self.fc2(hidden_states)
            return residual + hidden_states

    class MiMoAudioTokenizerEncoder(nn.Module):
        def __init__(self, config: MiMoAudioTokenizerConfig):
            super().__init__()
            self.config = config
            self.conv1 = nn.Conv1d(
                int(config.n_mels),
                int(config.d_model),
                int(config.kernel_size),
                padding=1,
                bias=True,
            )
            self.conv2 = nn.Conv1d(
                int(config.d_model),
                int(config.d_model),
                int(config.kernel_size),
                stride=int(config.stride_size),
                padding=1,
                bias=True,
            )
            attn_windows = []
            if bool(config.hybrid_attention):
                for idx in range(int(config.encoder_layers)):
                    if idx % int(config.swa_per_block) < int(config.swa_per_block) - 1:
                        attn_windows.append(tuple(config.encoder_attn_window_size))
                    else:
                        attn_windows.append((-1, -1))
            else:
                attn_windows = [tuple(config.encoder_attn_window_size)] * int(config.encoder_layers)
            self.layers = [
                MiMoAudioTokenizerLayer(config, window_size=attn_windows[idx])
                for idx in range(int(config.encoder_layers))
            ]
            self.layer_norm = nn.LayerNorm(int(config.d_model))
            self.down_sample_layer = (
                nn.Conv1d(
                    int(config.d_model),
                    int(config.d_model),
                    int(config.avg_pooler),
                    stride=int(config.avg_pooler),
                    bias=False,
                )
                if int(config.avg_pooler or 1) != 1
                else None
            )
            self.down_sample_norm = (
                nn.LayerNorm(int(config.d_model))
                if self.down_sample_layer is not None
                else None
            )
            self.quantizer = MiMoAudioResidualVectorQuantizer(
                [
                    mx.zeros((int(size), int(config.d_model)), dtype=mx.float32)
                    for size in list(config.codebook_size)[: int(config.num_quantizers)]
                ]
            )

        def get_output_length(self, mel_len):
            return self.config.get_output_length(mel_len)

        def get_features(self, input_features, input_lens):
            input_lens = mx.array(input_lens).astype(mx.int32)
            output_length = self.get_output_length(input_lens)
            hidden = _mimo_audio_unpack_segments(input_features, input_lens)
            hidden = _mimo_audio_activation(self.config.activation_function, self.conv1(hidden))
            hidden = _mimo_audio_activation(self.config.activation_function, self.conv2(hidden))
            packed = _mimo_audio_pack_segments(hidden, output_length)
            skip_hidden = None
            for idx, layer in enumerate(self.layers):
                packed = layer(packed, output_length)
                if self.config.encoder_skip_layer_id is not None and idx == int(self.config.encoder_skip_layer_id) - 1:
                    skip_hidden = packed
            if skip_hidden is not None:
                packed = packed + skip_hidden
            packed = self.layer_norm(packed)
            if self.down_sample_layer is not None:
                dense = _mimo_audio_unpack_segments(packed, output_length)
                avg = int(self.config.avg_pooler)
                tgt_len = int(dense.shape[1])
                if tgt_len % avg:
                    pad_len = avg - (tgt_len % avg)
                    dense = mx.concatenate(
                        [
                            dense,
                            mx.zeros((dense.shape[0], pad_len, dense.shape[2]), dtype=dense.dtype),
                        ],
                        axis=1,
                    )
                dense = _mimo_audio_activation(
                    self.config.activation_function,
                    self.down_sample_layer(dense),
                )
                output_length = self.config.get_code_length(input_lens)
                packed = _mimo_audio_pack_segments(dense, output_length)
                packed = self.down_sample_norm(packed)
            return packed, output_length

        def encode(
            self,
            input_features,
            input_lens=None,
            output_length=None,
            return_codes_only=False,
            n_q=None,
            use_quantizer=True,
        ):
            if input_lens is None:
                raise ValueError("MiMo audio tokenizer encode requires input_lens")
            hidden_states, output_length = self.get_features(input_features, input_lens)
            codes = None
            if use_quantizer and self.quantizer is not None:
                codes = self.quantizer.encode(hidden_states.astype(mx.float32), n_q=n_q)
                if return_codes_only:
                    return codes, output_length
                hidden_states = self.quantizer.decode(codes).astype(hidden_states.dtype)
            return hidden_states, hidden_states, output_length, codes

    class MiMoAudioTokenizer(nn.Module):
        def __init__(self, config: MiMoAudioTokenizerConfig):
            super().__init__()
            self.config = config
            self.sampling_rate = int(config.sampling_rate)
            self.encoder = MiMoAudioTokenizerEncoder(config)
            self.downsample_rate = int(config.hop_length * 2 * config.avg_pooler)

        def get_output_length(self, mel_len):
            return self.encoder.get_output_length(mel_len)

        def encode(self, mels, input_lens, use_quantizer=True, return_codes_only=False, n_q=None):
            return self.encoder.encode(
                mels,
                input_lens=input_lens,
                use_quantizer=use_quantizer,
                return_codes_only=return_codes_only,
                n_q=n_q,
            )

        def encode_audio_to_codes(self, mels, *, segment_size: int = 6000, n_q=None):
            plan = plan_mimo_audio_mel_segments(
                mels,
                config=self.config,
                segment_size=segment_size,
            )
            if not plan["code_lengths"]:
                return []
            codes, _ = self.encode(
                plan["input_features"],
                input_lens=plan["input_lens"],
                return_codes_only=True,
                n_q=n_q,
            )
            codes = mx.transpose(codes, (1, 0)).astype(mx.int32)
            result = []
            offset = 0
            for length in plan["code_lengths"]:
                result.append(codes[offset : offset + int(length)])
                offset += int(length)
            return result

    def load_mimo_audio_rvq_from_bundle(model_path: str | Path):
        """Load MiMo audio-tokenizer RVQ codebooks from a model bundle.

        This is the narrow, model-owned bridge from the preserved
        ``audio_tokenizer/model.safetensors`` sidecar into the MLX residual VQ
        runtime above. It intentionally stops at codebook loading: waveform/mel
        frontend and the 24-layer audio-tokenizer encoder are separate runtime
        components and must not be reported as implemented by this helper.
        """

        root = Path(model_path)
        audio_dir = root if root.name == "audio_tokenizer" else root / "audio_tokenizer"
        config_path = audio_dir / "config.json"
        weights_path = audio_dir / "model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"MiMo audio tokenizer config missing: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"MiMo audio tokenizer weights missing: {weights_path}")

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        try:
            num_quantizers = int(cfg["num_quantizers"])
        except Exception as exc:
            raise ValueError("MiMo audio tokenizer config missing num_quantizers") from exc
        codebook_size = cfg.get("codebook_size")
        if not isinstance(codebook_size, list) or len(codebook_size) < num_quantizers:
            raise ValueError(
                "MiMo audio tokenizer config codebook_size must list every quantizer"
            )

        try:
            from safetensors.torch import load_file as _torch_load_file

            raw_tensors = _torch_load_file(str(weights_path), device="cpu")

            def _load_tensor(key: str):
                if key not in raw_tensors:
                    raise KeyError(key)
                return mx.array(raw_tensors[key].float().cpu().numpy())

        except ImportError:
            from safetensors.numpy import load_file as _np_load_file

            raw_tensors = _np_load_file(str(weights_path))

            def _load_tensor(key: str):
                if key not in raw_tensors:
                    raise KeyError(key)
                return mx.array(raw_tensors[key])

        codebooks = []
        missing = []
        for idx in range(num_quantizers):
            key = f"encoder.quantizer.vq.layers.{idx}._codebook.embed"
            try:
                embed = _load_tensor(key)
            except KeyError:
                missing.append(key)
                continue
            expected_rows = int(codebook_size[idx])
            if int(embed.shape[0]) != expected_rows:
                raise ValueError(
                    f"MiMo audio tokenizer codebook {idx} shape mismatch: "
                    f"expected {expected_rows} rows, got {embed.shape[0]}"
                )
            codebooks.append(embed)
        if missing:
            preview = ", ".join(missing[:3])
            if len(missing) > 3:
                preview += f", ... ({len(missing)} missing)"
            raise KeyError(f"MiMo audio tokenizer codebook weights missing: {preview}")
        return MiMoAudioResidualVectorQuantizer(codebooks)

    def load_mimo_audio_tokenizer_from_bundle(model_path: str | Path):
        """Load executable MiMo audio tokenizer from ``audio_tokenizer/model.safetensors``."""

        root = Path(model_path)
        audio_dir = root if root.name == "audio_tokenizer" else root / "audio_tokenizer"
        config = MiMoAudioTokenizerConfig.from_bundle(audio_dir)
        tokenizer = MiMoAudioTokenizer(config)
        weights_path = audio_dir / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"MiMo audio tokenizer weights missing: {weights_path}")

        try:
            from safetensors.torch import load_file as _torch_load_file

            raw_tensors = _torch_load_file(str(weights_path), device="cpu")
            items = [
                (key, mx.array(value.float().cpu().numpy()))
                for key, value in raw_tensors.items()
                if "encoder.quantizer.vq.layers." not in key
            ]
        except ImportError:
            from safetensors.numpy import load_file as _np_load_file

            raw_tensors = _np_load_file(str(weights_path))
            items = [
                (key, mx.array(value))
                for key, value in raw_tensors.items()
                if "encoder.quantizer.vq.layers." not in key
            ]

        converted = []
        for key, value in items:
            if key in {
                "encoder.conv1.weight",
                "encoder.conv2.weight",
                "encoder.down_sample_layer.0.weight",
            } and value.ndim == 3:
                value = mx.transpose(value, (0, 2, 1))
            if key.startswith("encoder.down_sample_layer.0."):
                key = key.replace("encoder.down_sample_layer.0.", "encoder.down_sample_layer.", 1)
            converted.append((key, value))
        tokenizer.load_weights(converted, strict=False)
        tokenizer.encoder.quantizer = load_mimo_audio_rvq_from_bundle(audio_dir)
        return tokenizer

    def _parse_mimo_v2_int_list(value, length: int):
        if isinstance(value, str) and "-" in value:
            return [int(v) for v in value.split("-")]
        if isinstance(value, (list, tuple)):
            values = [int(v) for v in value]
            return values if len(values) == length else (values * length)[:length]
        return [int(value)] * length

    def _pad_and_group_mimo_v2_audio_codes(audio_codes, audio_channels: int, group_size: int):
        audio_codes = mx.array(audio_codes).astype(mx.int32)
        if audio_codes.ndim == 3:
            if int(audio_codes.shape[1]) != group_size:
                raise ValueError(
                    "MiMo-V2 grouped audio_codes group size mismatch: "
                    f"{audio_codes.shape[1]} != {group_size}"
                )
            return audio_codes[:, :, :audio_channels]
        if audio_codes.ndim != 2:
            raise ValueError(
                "MiMo-V2 audio_codes must be rank-2 [T, C] or rank-3 [G, group, C]; "
                f"got shape {audio_codes.shape}"
            )
        if int(audio_codes.shape[0]) <= 0:
            raise ValueError("MiMo-V2 audio_codes must contain at least one timestep")
        audio_codes = audio_codes[:, :audio_channels]
        t = int(audio_codes.shape[0])
        padded_t = ((t + group_size - 1) // group_size) * group_size
        if padded_t > t:
            pad = mx.broadcast_to(audio_codes[-1:, :], (padded_t - t, audio_codes.shape[1]))
            audio_codes = mx.concatenate([audio_codes, pad], axis=0)
        return mx.reshape(audio_codes, (padded_t // group_size, group_size, audio_codes.shape[1]))

    class MiMoAudioLocalMLP(nn.Module):
        def __init__(self, *, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class MiMoAudioLocalAttention(nn.Module):
        def __init__(
            self,
            *,
            hidden_size: int,
            num_heads: int,
            head_dim: int,
            rope_theta: float,
            partial_rotary_factor: float,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scaling = head_dim**-0.5
            self.rope_theta = float(rope_theta)
            self.rotary_dim = int(head_dim * float(partial_rotary_factor))
            self.rotary_dim = max(0, min(head_dim, self.rotary_dim))
            if self.rotary_dim % 2:
                self.rotary_dim -= 1
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
            self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
            self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        def _rotate_half(self, x):
            half = int(x.shape[-1]) // 2
            first = x[..., :half]
            second = x[..., half : half * 2]
            rest = x[..., half * 2 :]
            rotated = mx.concatenate([-second, first], axis=-1)
            return mx.concatenate([rotated, rest], axis=-1) if rest.shape[-1] else rotated

        def _apply_rope(self, q, k):
            if self.rotary_dim <= 0:
                return q, k
            seq_len = int(q.shape[-2])
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    mx.arange(0, self.rotary_dim, 2).astype(mx.float32)
                    / self.rotary_dim
                )
            )
            pos = mx.arange(seq_len).astype(mx.float32)
            freqs = pos[:, None] * inv_freq[None, :]
            emb = mx.concatenate([freqs, freqs], axis=-1)
            cos = mx.cos(emb)[None, None, :, :].astype(q.dtype)
            sin = mx.sin(emb)[None, None, :, :].astype(q.dtype)
            q_rope = q[..., : self.rotary_dim]
            k_rope = k[..., : self.rotary_dim]
            q = mx.concatenate(
                [
                    q_rope * cos + self._rotate_half(q_rope) * sin,
                    q[..., self.rotary_dim :],
                ],
                axis=-1,
            )
            k = mx.concatenate(
                [
                    k_rope * cos + self._rotate_half(k_rope) * sin,
                    k[..., self.rotary_dim :],
                ],
                axis=-1,
            )
            return q, k

        def __call__(self, hidden_states, *, is_causal: bool = False):
            hidden_states = mx.array(hidden_states)
            batch, seq_len, _ = hidden_states.shape
            q = mx.reshape(
                self.q_proj(hidden_states),
                (batch, seq_len, self.num_heads, self.head_dim),
            )
            k = mx.reshape(
                self.k_proj(hidden_states),
                (batch, seq_len, self.num_heads, self.head_dim),
            )
            v = mx.reshape(
                self.v_proj(hidden_states),
                (batch, seq_len, self.num_heads, self.head_dim),
            )
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            q, k = self._apply_rope(q, k)
            mask = None
            if is_causal:
                row = mx.arange(seq_len)[:, None]
                col = mx.arange(seq_len)[None, :]
                mask = mx.where(
                    col > row,
                    mx.full((seq_len, seq_len), -mx.inf),
                    mx.zeros((seq_len, seq_len)),
                )[None, None, :, :]
            attn = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scaling,
                mask=mask,
            )
            attn = mx.reshape(
                mx.transpose(attn, (0, 2, 1, 3)),
                (batch, seq_len, self.num_heads * self.head_dim),
            )
            return self.o_proj(attn)

    class MiMoAudioLocalBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            hidden_size = int(getattr(config, "input_local_dim", 1024) or 1024)
            num_heads = int(getattr(config, "input_local_attn_heads", 16) or 16)
            head_dim = int(getattr(config, "input_local_head_dim", 0) or 0)
            if head_dim <= 0:
                head_dim = max(1, hidden_size // max(1, num_heads))
            intermediate_size = int(
                getattr(config, "input_local_intermediate_size", hidden_size * 4)
                or hidden_size * 4
            )
            self.self_attn = MiMoAudioLocalAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                rope_theta=float(getattr(config, "rope_theta", 640000.0) or 640000.0),
                partial_rotary_factor=float(
                    getattr(config, "partial_rotary_factor", 1.0) or 1.0
                ),
            )
            self.mlp = MiMoAudioLocalMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)

        def __call__(self, hidden_states, *, is_causal: bool = False):
            hidden_states = hidden_states + self.self_attn(
                self.input_layernorm(hidden_states),
                is_causal=is_causal,
            )
            hidden_states = hidden_states + self.mlp(
                self.post_attention_layernorm(hidden_states)
            )
            return hidden_states

    class MiMoAudioLocalTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            hidden_size = int(getattr(config, "input_local_dim", 1024) or 1024)
            layers = int(getattr(config, "input_local_layers", 6) or 6)
            self.layers = [MiMoAudioLocalBlock(config) for _ in range(layers)]
            self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
            self.input_full_attention = bool(
                getattr(config, "input_full_attention", True)
            )

        def __call__(self, inputs_embeds):
            hidden_states = mx.array(inputs_embeds)
            is_causal = not self.input_full_attention
            for layer in self.layers:
                hidden_states = layer(hidden_states, is_causal=is_causal)
            return self.norm(hidden_states)

    class AudioModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config or AudioConfig()
            self.audio_channels = int(getattr(self.config, "audio_channels", 20) or 20)
            self.group_size = int(getattr(self.config, "group_size", 6) or 6)
            self.input_local_dim = int(
                getattr(self.config, "input_local_dim", None)
                or getattr(self.config, "hidden_size", 0)
                or 1280
            )
            self.out_hidden_size = int(
                getattr(self.config, "out_hidden_size", None) or 4096
            )
            self.input_full_attention = bool(
                getattr(self.config, "input_full_attention", True)
            )
            self.input_local_transformer = MiMoAudioLocalTransformer(self.config)
            proj_in = self.input_local_dim * self.group_size
            projection_layers = int(getattr(self.config, "projection_layers", 2) or 2)
            if projection_layers == 1:
                self.projection = nn.Linear(proj_in, self.out_hidden_size, bias=False)
            else:
                self.projection = AudioProjection(
                    input_size=proj_in,
                    hidden_size=proj_in * 4,
                    output_size=self.out_hidden_size,
                )

        def project_local_audio(self, audio_hidden):
            audio_hidden = mx.array(audio_hidden)
            if audio_hidden.ndim == 3:
                if int(audio_hidden.shape[1]) != self.group_size:
                    raise ValueError(
                        "MiMo-V2 audio local hidden group size mismatch: "
                        f"{audio_hidden.shape[1]} != {self.group_size}"
                    )
                audio_hidden = mx.reshape(audio_hidden, (audio_hidden.shape[0], -1))
            if audio_hidden.ndim != 2:
                raise ValueError(
                    "MiMo-V2 audio projection expects rank-2 or rank-3 audio hidden "
                    f"states; got shape {audio_hidden.shape}"
                )
            expected = self.input_local_dim * self.group_size
            if int(audio_hidden.shape[-1]) != expected:
                raise ValueError(
                    f"MiMo-V2 audio projection expected hidden size {expected}; "
                    f"got {audio_hidden.shape[-1]}"
                )
            return self.projection(audio_hidden)

        def _apply_speech_embeddings(self, audio_codes, speech_embeddings):
            if speech_embeddings is None:
                raise UnsupportedMediaModalityError(
                    "audio",
                    "MiMo-V2.5 audio_codes require speech_embeddings weights.",
                    family="mimo_v2",
                )
            grouped = _pad_and_group_mimo_v2_audio_codes(
                audio_codes,
                self.audio_channels,
                self.group_size,
            )
            if int(grouped.shape[-1]) < self.audio_channels:
                raise ValueError(
                    "MiMo-V2 audio_codes channel count mismatch: "
                    f"{grouped.shape[-1]} < {self.audio_channels}"
                )
            out = mx.zeros(
                (grouped.shape[0], self.group_size, self.input_local_dim),
                dtype=speech_embeddings[0].weight.dtype,
            )
            if len(speech_embeddings) < self.audio_channels:
                raise ValueError(
                    "MiMo-V2 speech embedding count mismatch: "
                    f"{len(speech_embeddings)} < {self.audio_channels}"
                )
            for idx in range(self.audio_channels):
                out = out + speech_embeddings[idx](grouped[:, :, idx])
            return out

        def process_audio_codes(self, audio_codes, speech_embeddings):
            audio_embs = self._apply_speech_embeddings(audio_codes, speech_embeddings)
            audio_hidden = self.input_local_transformer(audio_embs)
            return self.project_local_audio(audio_hidden)

        def __call__(
            self,
            speech_embeddings=None,
            audio_codes=None,
            audio_embeds=None,
            audio_hidden=None,
        ):
            if audio_embeds is not None:
                audio_embeds = mx.array(audio_embeds)
                if audio_embeds.ndim != 2:
                    raise ValueError(
                        "MiMo-V2 audio_embeds must be rank-2 [N, H]; "
                        f"got shape {audio_embeds.shape}"
                    )
                if int(audio_embeds.shape[-1]) == self.out_hidden_size:
                    return audio_embeds
                return self.project_local_audio(audio_embeds)
            if audio_hidden is not None:
                return self.project_local_audio(audio_hidden)
            if audio_codes is not None:
                return self.process_audio_codes(audio_codes, speech_embeddings)
            raise ValueError(
                "MiMo-V2 audio encoder requires audio_codes, audio_embeds, or audio_hidden"
            )

    class MiMoVisionPatchEmbed(nn.Module):
        def __init__(
            self,
            *,
            patch_size: int = 16,
            temporal_patch_size: int = 2,
            in_channels: int = 3,
            embed_dim: int = 1280,
        ):
            super().__init__()
            self.patch_size = patch_size
            self.temporal_patch_size = temporal_patch_size
            self.in_channels = in_channels
            self.embed_dim = embed_dim
            self.proj = nn.Linear(
                in_channels * temporal_patch_size * patch_size * patch_size,
                embed_dim,
                bias=False,
            )

        def __call__(self, hidden_states):
            hidden_states = mx.array(hidden_states)
            flat_patch_width = (
                self.in_channels
                * self.temporal_patch_size
                * self.patch_size
                * self.patch_size
            )
            if hidden_states.shape[-1] == flat_patch_width:
                return self.proj(hidden_states)
            expected = (
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
            try:
                patches = mx.reshape(hidden_states, expected)
            except Exception as exc:
                raise ValueError(
                    "MiMo-V2 vision patch embedding expected flattened patches "
                    f"of width {flat_patch_width} or reshapeable raw patches; "
                    f"got shape {hidden_states.shape}"
                ) from exc
            return self.proj(mx.reshape(patches, (patches.shape[0], flat_patch_width)))

    class MiMoVisionPatchMerger(nn.Module):
        def __init__(self, *, dim: int, context_dim: int, spatial_merge_size: int = 2):
            super().__init__()
            self.dim = dim
            self.context_dim = context_dim
            self.spatial_merge_size = spatial_merge_size
            self.hidden_size = context_dim * (spatial_merge_size**2)
            self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
            self.mlp = [
                nn.Linear(self.hidden_size, self.hidden_size),
                MiMoVisionGELU(),
                nn.Linear(self.hidden_size, dim),
            ]

        def __call__(self, x):
            x = mx.array(x)
            if x.ndim != 2 or int(x.shape[-1]) != self.context_dim:
                raise ValueError(
                    "MiMo-V2 vision merger expects rank-2 patch states with "
                    f"hidden size {self.context_dim}; got shape {x.shape}"
                )
            if int(x.shape[0]) % int(self.spatial_merge_size**2) != 0:
                raise ValueError(
                    "MiMo-V2 vision merger patch count must be divisible by "
                    f"{self.spatial_merge_size**2}; got {x.shape[0]}"
                )
            x = self.ln_q(x)
            x = mx.reshape(x, (-1, self.hidden_size))
            for layer in self.mlp:
                x = layer(x)
            return x

    class MiMoVisionGELU(nn.Module):
        def __call__(self, x):
            return nn.gelu(x)

    class MiMoVisionSwiGLUMLP(nn.Module):
        def __init__(self, *, dim: int, intermediate_dim: int, hidden_act: str = "silu"):
            super().__init__()
            self.gate_proj = nn.Linear(dim, intermediate_dim)
            self.up_proj = nn.Linear(dim, intermediate_dim)
            self.down_proj = nn.Linear(intermediate_dim, dim)
            self.hidden_act = hidden_act

        def _activate(self, x):
            if self.hidden_act in {"gelu", "gelu_pytorch_tanh"}:
                return nn.gelu(x)
            return x * mx.sigmoid(x)

        def __call__(self, x):
            return self.down_proj(self._activate(self.gate_proj(x)) * self.up_proj(x))

    class MiMoVisionAttention(nn.Module):
        def __init__(
            self,
            *,
            dim: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int,
            use_sinks: bool = False,
            window_size: int = -1,
        ):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.num_kv_groups = max(1, num_heads // max(1, num_kv_heads))
            self.scaling = head_dim**-0.5
            self.window_size = int(window_size)
            qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
            self.qkv = nn.Linear(dim, qkv_dim)
            self.proj = nn.Linear(num_heads * head_dim, dim)
            self.sinks = mx.zeros((num_heads,)) if use_sinks else None

        def _window_mask(self, seq_len: int):
            if self.window_size <= 0:
                return None
            row_idx = mx.arange(seq_len)[:, None]
            col_idx = mx.arange(seq_len)[None, :]
            mask = mx.where(
                mx.abs(row_idx - col_idx) > self.window_size,
                mx.full((seq_len, seq_len), -mx.inf),
                mx.zeros((seq_len, seq_len)),
            )
            return mask[None, None, :, :]

        def _rotate_half(self, x):
            half = int(x.shape[-1]) // 2
            first = x[..., :half]
            second = x[..., half : half * 2]
            rest = x[..., half * 2 :]
            rotated = mx.concatenate([-second, first], axis=-1)
            return mx.concatenate([rotated, rest], axis=-1) if rest.shape[-1] else rotated

        def _apply_rotary(self, q, k, position_embeddings):
            if position_embeddings is None:
                return q, k
            cos, sin = position_embeddings
            rope_dim = min(int(cos.shape[-1]), int(q.shape[-1]))
            cos = cos[:, :rope_dim][None, None, :, :]
            sin = sin[:, :rope_dim][None, None, :, :]
            q_rope = q[..., :rope_dim]
            k_rope = k[..., :rope_dim]
            q = mx.concatenate(
                [q_rope * cos + self._rotate_half(q_rope) * sin, q[..., rope_dim:]],
                axis=-1,
            )
            k = mx.concatenate(
                [k_rope * cos + self._rotate_half(k_rope) * sin, k[..., rope_dim:]],
                axis=-1,
            )
            return q, k

        def __call__(
            self,
            hidden_states,
            full_attn: bool = False,
            position_embeddings=None,
        ):
            hidden_states = mx.array(hidden_states)
            if hidden_states.ndim != 2 or int(hidden_states.shape[-1]) != self.dim:
                raise ValueError(
                    "MiMo-V2 vision attention expects rank-2 hidden states with "
                    f"hidden size {self.dim}; got shape {hidden_states.shape}"
                )
            seq_len = int(hidden_states.shape[0])
            qkv = self.qkv(hidden_states)
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_kv_heads * self.head_dim
            q = mx.reshape(qkv[:, :q_dim], (seq_len, self.num_heads, self.head_dim))
            k = mx.reshape(
                qkv[:, q_dim : q_dim + kv_dim],
                (seq_len, self.num_kv_heads, self.head_dim),
            )
            v = mx.reshape(qkv[:, q_dim + kv_dim :], (seq_len, self.num_kv_heads, self.head_dim))
            q = mx.transpose(q[None, :, :, :], (0, 2, 1, 3))
            k = mx.transpose(k[None, :, :, :], (0, 2, 1, 3))
            v = mx.transpose(v[None, :, :, :], (0, 2, 1, 3))
            q, k = self._apply_rotary(q, k, position_embeddings)
            if self.num_kv_groups > 1:
                k = mx.repeat(k, self.num_kv_groups, axis=1)
                v = mx.repeat(v, self.num_kv_groups, axis=1)
            mask = None if full_attn else self._window_mask(seq_len)
            if self.sinks is not None:
                sink_bias = mx.zeros((1, self.num_heads, seq_len, seq_len))
                sink_bias = sink_bias + self.sinks[None, :, None, None] * (
                    mx.arange(seq_len)[None, None, None, :] == 0
                )
                mask = sink_bias if mask is None else mask + sink_bias
            attn_output = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scaling,
                mask=mask,
            )
            attn_output = mx.reshape(
                mx.transpose(attn_output, (0, 2, 1, 3)),
                (seq_len, self.num_heads * self.head_dim),
            )
            return self.proj(attn_output)

    class MiMoVisionBlock(nn.Module):
        def __init__(
            self,
            *,
            dim: int,
            intermediate_dim: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int,
            hidden_act: str = "silu",
            rms_norm_eps: float = 1e-6,
            use_sinks: bool = False,
            window_size: int = -1,
        ):
            super().__init__()
            self.norm1 = nn.RMSNorm(dim, eps=rms_norm_eps)
            self.norm2 = nn.RMSNorm(dim, eps=rms_norm_eps)
            self.attn = MiMoVisionAttention(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                use_sinks=use_sinks,
                window_size=window_size,
            )
            self.mlp = MiMoVisionSwiGLUMLP(
                dim=dim,
                intermediate_dim=intermediate_dim,
                hidden_act=hidden_act,
            )

        def __call__(
            self,
            hidden_states,
            full_attn: bool = False,
            position_embeddings=None,
        ):
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                full_attn=full_attn,
                position_embeddings=position_embeddings,
            )
            return hidden_states + self.mlp(self.norm2(hidden_states))

    def _mimo_v2_flatten_modal_embeds(modal_embeds):
        arr = mx.array(modal_embeds)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3:
            arr = mx.reshape(arr, (-1, arr.shape[-1]))
        elif arr.ndim != 2:
            raise ValueError(
                f"MiMo-V2 modal embeddings must be rank 1, 2, or 3; got shape {arr.shape}"
            )
        return arr

    def _mimo_v2_replace_modal_embeddings(
        *,
        input_ids,
        inputs_embeds,
        token_id,
        modal_embeds,
        modality: str,
    ):
        if modal_embeds is None:
            return inputs_embeds
        if token_id is None:
            raise ValueError(f"MiMo-V2 {modality} token id is missing from config")

        ids = mx.array(input_ids)
        embeds = mx.array(inputs_embeds)
        if ids.ndim == 1:
            ids = ids[None, :]
        if embeds.ndim == 2:
            embeds = embeds[None, :, :]
        if ids.ndim != 2 or embeds.ndim != 3:
            raise ValueError(
                f"MiMo-V2 {modality} splice expects input_ids rank 1/2 and "
                f"inputs_embeds rank 2/3; got {ids.shape} and {embeds.shape}"
            )
        if ids.shape[0] != embeds.shape[0] or ids.shape[1] != embeds.shape[1]:
            raise ValueError(
                f"MiMo-V2 {modality} splice shape mismatch: input_ids {ids.shape}, "
                f"inputs_embeds {embeds.shape}"
            )

        flat_modal = _mimo_v2_flatten_modal_embeds(modal_embeds)
        hidden = embeds.shape[-1]
        if flat_modal.shape[-1] != hidden:
            raise ValueError(
                f"MiMo-V2 {modality} embedding hidden size mismatch: "
                f"{flat_modal.shape[-1]} != {hidden}"
            )

        ids_py = ids.tolist()
        positions: list[tuple[int, int]] = []
        for batch_idx, row in enumerate(ids_py):
            for token_idx, value in enumerate(row):
                if int(value) == int(token_id):
                    positions.append((batch_idx, token_idx))

        if len(positions) != flat_modal.shape[0]:
            raise ValueError(
                f"MiMo-V2 {modality} token count {len(positions)} does not match "
                f"provided embeddings {flat_modal.shape[0]}"
            )
        if not positions:
            return embeds

        modal_index = 0
        batch_rows = []
        for batch_idx, row in enumerate(ids_py):
            row_parts = []
            last = 0
            row_positions = [pos for b, pos in positions if b == batch_idx]
            for pos in row_positions:
                if pos > last:
                    row_parts.append(embeds[batch_idx : batch_idx + 1, last:pos, :])
                row_parts.append(flat_modal[modal_index : modal_index + 1][None, :, :])
                modal_index += 1
                last = pos + 1
            if last < len(row):
                row_parts.append(embeds[batch_idx : batch_idx + 1, last:, :])
            batch_rows.append(mx.concatenate(row_parts, axis=1))
        return mx.concatenate(batch_rows, axis=0)

    class Model(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.language_model = _MiMoV2LanguageModelCompat(TextModel(config.text_config))
            self._mimo_v2_quantized_for_load = False
            self._mimo_v2_pending_qkv_affine: dict[str, dict[str, Any]] = {}
            self._mimo_v2_media_weights: dict[str, Any] = {}
            self._mimo_v2_media_weight_counts = {
                "visual": 0,
                "audio_encoder": 0,
                "speech_embeddings": 0,
            }
            self._mimo_v2_bind_media_weights = bool(
                getattr(config, "media_runtime_wired", False)
                and (
                    isinstance(getattr(config, "vision_config", None), VisionConfig)
                    or isinstance(getattr(config, "audio_config", None), AudioConfig)
                )
            )
            self.visual = (
                VisionModel(config.vision_config)
                if self._mimo_v2_bind_media_weights
                else None
            )
            self.audio_encoder = (
                AudioModel(config.audio_config)
                if self._mimo_v2_bind_media_weights
                else None
            )
            self.speech_embeddings = (
                [
                    nn.Embedding(vocab_size, self.audio_encoder.input_local_dim)
                    for vocab_size in _parse_mimo_v2_int_list(
                        getattr(config.audio_config, "speech_vocab_size", 1280),
                        self.audio_encoder.audio_channels,
                    )
                ]
                if self._mimo_v2_bind_media_weights and self.audio_encoder is not None
                else None
            )

        @property
        def layers(self):
            return self.language_model.layers

        def get_input_embeddings(
            self,
            input_ids=None,
            pixel_values=None,
            image_grid_thw=None,
            image_embeds=None,
            video_pixel_values=None,
            video_grid_thw=None,
            video_embeds=None,
            audio_codes=None,
            audio_embeds=None,
            **kwargs,
        ):
            from mlx_vlm.models.base import InputEmbeddingsFeatures

            if pixel_values is not None:
                if self.visual is None:
                    raise UnsupportedMediaModalityError(
                        "vision",
                        "MiMo-V2.5 JANG_2L vision weights are preserved, but this "
                        "bundle is configured for text-only runtime.",
                        family="mimo_v2",
                    )
                if image_grid_thw is None:
                    image_grid_thw = kwargs.get("grid_thw")
                if image_grid_thw is None:
                    raise ValueError("MiMo-V2 image requests require image_grid_thw")
                image_embeds = self.visual(
                    pixel_values=pixel_values,
                    grid_thw=image_grid_thw,
                )
            if video_pixel_values is not None:
                if self.visual is None:
                    raise UnsupportedMediaModalityError(
                        "video",
                        "MiMo-V2.5 JANG_2L video weights are preserved, but this "
                        "bundle is configured for text-only runtime.",
                        family="mimo_v2",
                    )
                if video_grid_thw is None:
                    raise ValueError("MiMo-V2 video requests require video_grid_thw")
                video_embeds = self.visual(
                    pixel_values=video_pixel_values,
                    grid_thw=video_grid_thw,
                )
            if audio_codes is not None:
                if self.audio_encoder is None or self.speech_embeddings is None:
                    raise UnsupportedMediaModalityError(
                        "audio",
                        "MiMo-V2.5 JANG_2L audio code runtime requires media-enabled metadata.",
                        family="mimo_v2",
                    )
                audio_embeds = self.audio_encoder(
                    speech_embeddings=self.speech_embeddings,
                    audio_codes=audio_codes,
                )
            if audio_embeds is not None:
                audio_arr = mx.array(audio_embeds)
                if (
                    self.audio_encoder is not None
                    and audio_arr.ndim == 2
                    and int(audio_arr.shape[-1])
                    != int(getattr(self.config.text_config, "hidden_size", 4096))
                ):
                    audio_embeds = self.audio_encoder(audio_embeds=audio_arr)
                else:
                    audio_embeds = audio_arr
            if input_ids is None:
                raise ValueError("MiMo-V2 get_input_embeddings requires input_ids")
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            if image_embeds is not None:
                inputs_embeds = _mimo_v2_replace_modal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    token_id=getattr(self.config, "image_token_id", None),
                    modal_embeds=image_embeds,
                    modality="image",
                )
            if video_embeds is not None:
                inputs_embeds = _mimo_v2_replace_modal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    token_id=getattr(self.config, "video_token_id", None),
                    modal_embeds=video_embeds,
                    modality="video",
                )
            if audio_embeds is not None:
                inputs_embeds = _mimo_v2_replace_modal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    token_id=getattr(self.config, "audio_token_id", None),
                    modal_embeds=audio_embeds,
                    modality="audio",
                )
            return InputEmbeddingsFeatures(
                inputs_embeds=inputs_embeds
            )

        def __call__(
            self,
            input_ids,
            pixel_values=None,
            image_grid_thw=None,
            image_embeds=None,
            video_pixel_values=None,
            video_grid_thw=None,
            video_embeds=None,
            audio_codes=None,
            audio_embeds=None,
            inputs_embeds=_INPUTS_EMBEDS_UNSET,
            mask=None,
            cache=None,
            **kwargs,
        ):
            if pixel_values is not None:
                inputs_embeds = self.get_input_embeddings(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                ).inputs_embeds
                input_ids = None
            if any(
                item is not None
                for item in (
                    image_embeds,
                    video_pixel_values,
                    video_embeds,
                    audio_codes,
                    audio_embeds,
                )
            ):
                inputs_embeds = self.get_input_embeddings(
                    input_ids=input_ids,
                    image_embeds=image_embeds,
                    video_pixel_values=video_pixel_values,
                    video_grid_thw=video_grid_thw,
                    video_embeds=video_embeds,
                    audio_codes=audio_codes,
                    audio_embeds=audio_embeds,
                ).inputs_embeds
                input_ids = None
            if inputs_embeds is not _INPUTS_EMBEDS_UNSET:
                return self.language_model(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    cache=cache,
                    mask=mask,
                    **kwargs,
                )
            return self.language_model(input_ids, cache=cache, mask=mask, **kwargs)

        def _bind_mimo_v2_media_weight(self, key, value):
            """Bind preserved MiMo media tensors when media runtime is enabled.

            Current text-only JANG_2L/JANGTQ_2 bundles intentionally advertise
            `weights_preserved_text_runtime`, so they should not retain the
            visual/audio tensor payload in memory. A future media-enabled MiMo
            config must not silently drop those tensors: this store is the
            load_weights boundary that real visual/audio modules consume before
            the forward bridge can be considered wired.
            """

            if key.startswith("visual."):
                group = "visual"
            elif key.startswith("audio_encoder."):
                group = "audio_encoder"
            elif key.startswith("speech_embeddings."):
                group = "speech_embeddings"
            else:
                return False
            if not self._mimo_v2_bind_media_weights:
                return True
            self._mimo_v2_media_weights[key] = value
            self._mimo_v2_media_weight_counts[group] += 1
            return True

        def _apply_mimo_v2_media_weights(self):
            if not self._mimo_v2_bind_media_weights or not self._mimo_v2_media_weights:
                return 0
            assigned = 0
            failures = []
            for key, value in self._mimo_v2_media_weights.items():
                try:
                    _mimo_v2_assign_weight(self, key, value)
                    assigned += 1
                except Exception as exc:
                    failures.append(f"{key}: {type(exc).__name__}: {exc}")
            if failures:
                preview = "; ".join(failures[:8])
                raise RuntimeError(
                    "MiMo-V2 media-enabled load could not bind preserved media "
                    f"weights ({len(failures)} failures): {preview}"
                )
            return assigned

        def load_weights(self, weights, strict=True):
            filtered = {}
            for key, value in weights:
                if key.startswith("model.mtp."):
                    continue
                if key.startswith(("visual.", "audio_encoder.", "speech_embeddings.")):
                    self._bind_mimo_v2_media_weight(key, value)
                    continue
                filtered[key] = value
            if self._mimo_v2_bind_media_weights and any(
                self._mimo_v2_media_weight_counts.values()
            ):
                logger.info(
                    "MiMo-V2 load bound preserved media weights: visual=%d audio_encoder=%d speech_embeddings=%d",
                    self._mimo_v2_media_weight_counts["visual"],
                    self._mimo_v2_media_weight_counts["audio_encoder"],
                    self._mimo_v2_media_weight_counts["speech_embeddings"],
                )
                assigned_media = self._apply_mimo_v2_media_weights()
                logger.info(
                    "MiMo-V2 load assigned %d preserved media tensors to runtime modules",
                    assigned_media,
                )

            if hasattr(self.language_model, "sanitize"):
                filtered = self.language_model.sanitize(filtered)

            for key, value in filtered.items():
                if not (
                    ".self_attn.qkv_proj." in key
                    or ".mlp.gate_proj." in key
                    or ".mlp.up_proj." in key
                    or ".mlp.down_proj." in key
                ):
                    continue
                for suffix in ("weight", "scales", "biases"):
                    marker = f".{suffix}"
                    if key.endswith(marker):
                        base = key[: -len(marker)]
                        self._mimo_v2_pending_qkv_affine.setdefault(base, {})[
                            suffix
                        ] = value
                        break

            quantization = getattr(self.config.text_config, "quantization", None)
            if isinstance(quantization, dict) and not self._mimo_v2_quantized_for_load:
                quantized_count = _quantize_mimo_v2_runtime_modules(
                    self.language_model.inner,
                    filtered,
                    quantization,
                )
                logger.info("MiMo-V2 load quantized %d runtime modules", quantized_count)
                self._mimo_v2_quantized_for_load = True

            tq_count = _install_mimo_v2_prestacked_jangtq_modules(
                self.language_model.inner,
                filtered,
                seed=int(getattr(self.config.text_config, "mxtq_seed", 42) or 42),
            )
            if tq_count:
                logger.info(
                    "MiMo-V2 load installed %d prestacked JANGTQ routed modules",
                    tq_count,
                )
                strict_load_weights = {}
                for key, value in filtered.items():
                    if key.endswith(".tq_packed"):
                        strict_load_weights[f"{key[: -len('.tq_packed')]}.packed"] = value
                    elif key.endswith(".tq_norms"):
                        strict_load_weights[f"{key[: -len('.tq_norms')]}.norms"] = value
                    elif (
                        key.endswith(".tq_bits")
                        or key.startswith("codebook.")
                        or key.startswith("signs.")
                    ):
                        continue
                    else:
                        strict_load_weights[key] = value
                filtered = strict_load_weights

            result = self.language_model.load_weights(list(filtered.items()), strict=strict)
            if isinstance(quantization, dict):
                try:
                    from vmlx_engine.utils.jang_loader import (
                        _upgrade_modules_with_uint32_weights,
                    )

                    upgraded_count = _upgrade_modules_with_uint32_weights(
                        self.language_model.inner,
                        default_bits=int(quantization.get("bits") or 8),
                        default_group_size=int(quantization.get("group_size") or 64),
                        default_mode=str(quantization.get("mode") or "affine"),
                    )
                    if upgraded_count:
                        logger.info(
                            "MiMo-V2 load upgraded %d packed affine runtime modules after load",
                            upgraded_count,
                        )
                except Exception:
                    logger.exception("MiMo-V2 post-load packed affine upgrade failed")
                pending_qkv_count = _install_mimo_v2_pending_qkv_affine_modules(
                    self.language_model.inner,
                    self._mimo_v2_pending_qkv_affine,
                    quantization,
                )
                if pending_qkv_count:
                    logger.info(
                        "MiMo-V2 load installed %d split-shard affine qkv modules",
                        pending_qkv_count,
                    )
                pending_mlp_count = _install_mimo_v2_pending_dense_mlp_affine_modules(
                    self.language_model.inner,
                    self._mimo_v2_pending_qkv_affine,
                    quantization,
                )
                if pending_mlp_count:
                    logger.info(
                        "MiMo-V2 load installed %d split-shard affine dense MLP modules",
                        pending_mlp_count,
                    )
                qkv_upgrade_count = _upgrade_mimo_v2_loaded_qkv_affine_modules(
                    self.language_model.inner,
                    quantization,
                )
                if qkv_upgrade_count:
                    logger.info(
                        "MiMo-V2 load upgraded %d packed affine qkv modules after load",
                        qkv_upgrade_count,
                    )
                hotspot_count = _quantize_mimo_v2_passthrough_decode_hotspots(
                    self.language_model.inner,
                    quantization,
                )
                if hotspot_count:
                    logger.info(
                        "MiMo-V2 load runtime-quantized %d passthrough decode hotspot modules",
                        hotspot_count,
                    )
            return result

        def sanitize(self, weights):
            media_weights = {
                key: value
                for key, value in weights.items()
                if key.startswith(("visual.", "audio_encoder.", "speech_embeddings."))
            }
            text_weights = {
                key: value
                for key, value in weights.items()
                if key not in media_weights
            }
            sanitized = self.language_model.sanitize(text_weights)
            sanitized.update(media_weights)
            return sanitized

    module = types.ModuleType(module_name)
    module.Model = Model
    module.ModelConfig = ModelConfig
    module.TextConfig = TextConfig
    module.VisionConfig = VisionConfig
    module.AudioConfig = AudioConfig
    module.LanguageModel = TextModel
    module.VisionModel = VisionModel
    module.MiMoVisionPatchEmbed = MiMoVisionPatchEmbed
    module.MiMoVisionPatchMerger = MiMoVisionPatchMerger
    module.MiMoVisionBlock = MiMoVisionBlock
    module.AudioModel = AudioModel
    module.MiMoAudioTokenizerConfig = MiMoAudioTokenizerConfig
    module.MiMoAudioTokenizer = MiMoAudioTokenizer
    module.MiMoAudioTokenizerEncoder = MiMoAudioTokenizerEncoder
    module.MiMoAudioEuclideanCodebook = MiMoAudioEuclideanCodebook
    module.MiMoAudioResidualVectorQuantizer = MiMoAudioResidualVectorQuantizer
    module.load_mimo_audio_rvq_from_bundle = load_mimo_audio_rvq_from_bundle
    module.load_mimo_audio_tokenizer_from_bundle = load_mimo_audio_tokenizer_from_bundle
    module.plan_mimo_audio_mel_segments = plan_mimo_audio_mel_segments
    module._vmlx_mimo_v2_runtime_registered = True
    module.__file__ = getattr(text_runtime, "__file__", module_name)
    sys.modules[module_name] = module


def _vlm_stream():
    """Create one stream for simple mlx-vlm generation paths.

    The batched VLM scheduler already pins MLX work to a dedicated stream.
    SimpleEngine's direct `mlx_vlm.stream_generate` path must do the same or
    JANGTQ/VL kernels can later materialize on a thread with no MLX stream.
    """
    global _VLM_STREAM
    if _VLM_STREAM is None:
        try:
            import mlx.core as mx

            _VLM_STREAM = mx.new_stream(mx.default_device())
        except Exception:
            try:
                import mlx.core as mx

                _VLM_STREAM = mx.default_stream(mx.default_device())
            except Exception:
                _VLM_STREAM = False
    return None if _VLM_STREAM is False else _VLM_STREAM


def reset_vlm_stream() -> None:
    """Clear the direct mlx-vlm stream handle after model teardown.

    ``_VLM_STREAM`` is owned by the worker thread that first runs direct
    ``mlx_vlm.stream_generate``. Deep sleep, wake, model switch, and direct
    SimpleEngine teardown can replace that worker; retaining the old stream
    lets a later request enter ``mx.stream(old_stream)`` from a different
    thread and raises ``RuntimeError: There is no Stream(gpu, N) in current
    thread``.
    """
    global _VLM_STREAM
    _VLM_STREAM = None


def _set_vlm_inference_mode(model: Any) -> None:
    """Put mlx-vlm model copies in eval mode so inference kernels stay enabled."""
    seen: set[int] = set()
    stack = [
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        eval_fn = getattr(candidate, "eval", None)
        if callable(eval_fn):
            try:
                eval_fn()
            except Exception:
                pass
        if hasattr(candidate, "training"):
            try:
                setattr(candidate, "training", False)
            except Exception:
                pass
        children_fn = getattr(candidate, "children", None)
        if callable(children_fn):
            try:
                children = children_fn()
                if isinstance(children, dict):
                    stack.extend(children.values())
                else:
                    stack.extend(children)
            except Exception:
                pass
        for attr in ("inner", "language_model", "model", "layers"):
            value = getattr(candidate, attr, None)
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif value is not None:
                stack.append(value)


class _MaybeVLMStream:
    __slots__ = ("_cm",)

    def __init__(self):
        self._cm = None
        stream = _vlm_stream()
        if stream is not None:
            try:
                import mlx.core as mx

                self._cm = mx.stream(stream)
            except Exception:
                self._cm = None

    def __enter__(self):
        if self._cm is not None:
            self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        if self._cm is not None:
            return self._cm.__exit__(*exc)
        return False


def _prompt_cache_state_from_entry(cache_entry: Any) -> Any | None:
    """Build mlx-vlm PromptCacheState from a vMLX MLLM prefix entry."""
    if cache_entry is None:
        return None
    kv_cache = getattr(cache_entry, "kv_cache", None)
    token_ids = getattr(cache_entry, "token_ids", None)
    if kv_cache is None or not token_ids:
        return None
    # Legacy MLLMPrefixCacheManager.store_cache() writes dummy all-zero token IDs
    # when the caller only knows a token count. mlx-vlm PromptCacheState uses
    # token_ids as the authoritative trim boundary, so dummy IDs would turn a
    # stale cache into a fake prefix hit.
    if all(token_id == 0 for token_id in token_ids):
        return None
    if len(token_ids) <= 1:
        return None
    restore_token_ids = list(token_ids[:-1])
    restore_len = len(restore_token_ids)
    try:
        kv_cache = _copy_prompt_cache_to_prompt_boundary(kv_cache, restore_len)
    except Exception:
        return None
    if not kv_cache:
        return None
    try:
        from mlx_vlm.generate import PromptCacheState
    except Exception:
        return None

    state = PromptCacheState()
    try:
        from ..utils.mlx_vlm_compat import _vmlx_wrap_rank3_prompt_cache_for_mlx_vlm

        kv_cache = _vmlx_wrap_rank3_prompt_cache_for_mlx_vlm(kv_cache)
    except Exception:
        pass
    state.cache = kv_cache
    state.token_ids = restore_token_ids
    return state


def _media_expanded_token_ids_for_cache(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    images: list[Any] | None = None,
    audio: list[Any] | None = None,
    video: list[Any] | None = None,
    resize_shape: Any = None,
) -> list[int] | None:
    """Return mlx-vlm's media-expanded input IDs for VLM prefix-cache keys."""
    if not images and not audio and not video:
        return None
    try:
        from mlx_vlm.generate import normalize_resize_shape, prepare_inputs
    except Exception:
        return None
    try:
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "") or "")
        image_token_index = getattr(config, "image_token_index", None)
        add_special_tokens = (
            getattr(processor, "chat_template", None) is None
            if model_type in {"gemma3", "gemma3n", "gemma4"}
            else True
        )
        inputs = prepare_inputs(
            processor,
            images=images if images else None,
            audio=audio if audio else None,
            videos=video if video else None,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=normalize_resize_shape(resize_shape),
            add_special_tokens=add_special_tokens,
        )
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            return None
        return [int(x) for x in input_ids.flatten().tolist()]
    except Exception as exc:
        logger.debug("Media-expanded VLM cache tokenization failed: %s", exc)
        return None


def _supports_simple_media_prompt_cache(model: Any) -> bool:
    """Whether simple mlx-vlm media PromptCacheState restore is safe."""
    language_model = getattr(model, "language_model", None)
    inner_model = getattr(language_model, "model", None)
    layers = getattr(inner_model, "layers", None)
    if not layers:
        return True
    for layer in layers:
        if hasattr(layer, "linear_attn") or hasattr(layer, "mamba") or hasattr(layer, "ssm"):
            return False
    return True


def _copy_prompt_cache_to_prompt_boundary(
    prompt_cache: list[Any], prompt_tokens_count: int
) -> list[Any]:
    """Copy an mlx-vlm prompt cache and trim generated-token tails."""
    import copy

    import mlx.core as mx

    cache_to_store = []
    prompt_tokens_count = max(int(prompt_tokens_count or 0), 0)
    for layer_cache in prompt_cache:
        new_cache = copy.copy(layer_cache)
        copied = False
        if hasattr(layer_cache, "state"):
            state = layer_cache.state
            if state is not None and len(state) >= 2 and state[0] is not None:
                keys = mx.array(state[0])
                values = mx.array(state[1])
                cache_axis = 2 if len(keys.shape) >= 4 else 1 if len(keys.shape) == 3 else None
                cache_len = keys.shape[cache_axis] if cache_axis is not None else None
                target_len = prompt_tokens_count
                if cache_len is not None:
                    target_len = min(prompt_tokens_count, cache_len)
                    if cache_len > target_len:
                        if cache_axis == 2:
                            keys = keys[:, :, :target_len, :]
                            values = values[:, :, :target_len, :]
                        else:
                            keys = keys[:, :target_len, :]
                            values = values[:, :target_len, :]
                new_cache.keys = keys
                new_cache.values = values
                if len(state) >= 3:
                    offset = state[2]
                    new_cache.offset = target_len if offset is None else min(offset, target_len)
                elif hasattr(layer_cache, "offset"):
                    offset = layer_cache.offset
                    new_cache.offset = target_len if offset is None else min(offset, target_len)
                copied = True
        if not copied:
            keys = getattr(layer_cache, "keys", None)
            values = getattr(layer_cache, "values", None)
            if keys is not None and values is not None:
                keys = mx.array(keys)
                values = mx.array(values)
                cache_axis = 2 if len(keys.shape) >= 4 else 1 if len(keys.shape) == 3 else None
                cache_len = keys.shape[cache_axis] if cache_axis is not None else None
                target_len = prompt_tokens_count
                if cache_len is not None:
                    target_len = min(prompt_tokens_count, cache_len)
                    if cache_len > target_len:
                        if cache_axis == 2:
                            keys = keys[:, :, :target_len, :]
                            values = values[:, :, :target_len, :]
                        else:
                            keys = keys[:, :target_len, :]
                            values = values[:, :target_len, :]
                new_cache.keys = keys
                new_cache.values = values
                if hasattr(layer_cache, "offset"):
                    offset = layer_cache.offset
                    new_cache.offset = target_len if offset is None else min(offset, target_len)
        cache_to_store.append(new_cache)
    return cache_to_store


class TempFileManager:
    """Thread-safe manager for tracking and cleaning up temporary files."""

    def __init__(self):
        self._files: set[str] = set()
        self._lock = threading.Lock()
        atexit.register(self.cleanup_all)

    def register(self, path: str) -> str:
        """Register a temp file for tracking. Returns the path for convenience."""
        with self._lock:
            self._files.add(path)
        return path

    def cleanup(self, path: str) -> bool:
        """Clean up a specific temp file. Returns True if successful."""
        try:
            if os.path.exists(path):
                os.unlink(path)
                with self._lock:
                    self._files.discard(path)
                logger.debug(f"Cleaned up temp file: {path}")
                return True
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")
        return False

    def cleanup_all(self) -> int:
        """Clean up all tracked temp files. Returns count of cleaned files."""
        with self._lock:
            files_to_clean = list(self._files)
            self._files.clear()

        cleaned = 0
        for path in files_to_clean:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    cleaned += 1
            except OSError:
                pass

        # Do not log from atexit cleanup: logging streams may already be closed
        # by pytest or the app shutdown path, which makes logging print noisy
        # "I/O operation on closed file" diagnostics despite successful cleanup.
        return cleaned


# Global temp file manager
_temp_manager = TempFileManager()



# Video processing constants
FRAME_FACTOR = 2  # Frames must be divisible by this
DEFAULT_FPS = 2.0  # Default frames per second for video
MIN_FRAMES = 4
MAX_FRAMES = 128  # Practical limit for most MLLMs
IMAGE_FACTOR = 28  # For smart resize

# Security: File size limits (in bytes)
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB max for images
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB max for videos
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB max for audio
MAX_BASE64_IMAGE_LENGTH = 30 * 1024 * 1024  # 30 MB base64 string (~22 MB decoded)
MAX_BASE64_VIDEO_LENGTH = 700 * 1024 * 1024  # 700 MB base64 string (~500 MB decoded)
MAX_BASE64_AUDIO_LENGTH = 70 * 1024 * 1024  # 70 MB base64 string (~50 MB decoded)


class FileSizeExceededError(Exception):
    """Raised when a downloaded file exceeds the size limit."""

    pass


@dataclass
class MultimodalInput:
    """Input for multimodal generation."""

    prompt: str
    images: list[str] = field(default_factory=list)  # Paths, URLs, or base64
    videos: list[str] = field(default_factory=list)  # Paths
    audio: list[str] = field(default_factory=list)  # Paths


@dataclass
class MLLMOutput:
    """Output from multimodal language model."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def is_base64_image(s: str) -> bool:
    """Check if string is base64-encoded image data."""
    return s.startswith("data:image/") or (
        len(s) > 100 and not s.startswith(("http://", "https://", "/"))
    )


def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://"))


def is_base64_video(s: str) -> bool:
    """Check if string is base64-encoded video data."""
    return s.startswith("data:video/")


def is_base64_audio(s: str) -> bool:
    """Check if string is base64-encoded audio data."""
    return s.startswith("data:audio/")


def decode_base64_image(
    base64_string: str, max_length: int = MAX_BASE64_IMAGE_LENGTH
) -> bytes:
    """
    Decode base64 image to bytes.

    Args:
        base64_string: Base64 encoded image (optionally with data URL prefix)
        max_length: Maximum allowed length of base64 string

    Returns:
        Decoded image bytes

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 image data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
    if base64_string.startswith("data:"):
        # Extract the base64 part after the comma
        _, data = base64_string.split(",", 1)
        return base64.b64decode(data)
    return base64.b64decode(base64_string)


def download_image(url: str, timeout: int = 30, max_size: int = MAX_IMAGE_SIZE) -> str:
    """
    Download image from URL and return local path.

    Args:
        url: Image URL
        timeout: Download timeout in seconds
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded image

    Raises:
        FileSizeExceededError: If image exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    # First, make a HEAD request to check Content-Length
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "png" in content_type:
        ext = ".png"
    elif "gif" in content_type:
        ext = ".gif"
    elif "webp" in content_type:
        ext = ".webp"
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = Path(path).suffix or ".jpg"

    # Save to temp file with size checking during download
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Image at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    return _temp_manager.register(temp_file.name)


def download_video(url: str, timeout: int = 120, max_size: int = MAX_VIDEO_SIZE) -> str:
    """
    Download video from URL and return local path.

    Args:
        url: Video URL (http/https)
        timeout: Download timeout in seconds (default 120s for larger videos)
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded video

    Raises:
        FileSizeExceededError: If video exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    logger.info(f"Downloading video from: {url}")

    # First, make a HEAD request to check Content-Length
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Video at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Video at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "mp4" in content_type:
        ext = ".mp4"
    elif "webm" in content_type:
        ext = ".webm"
    elif "avi" in content_type:
        ext = ".avi"
    elif "mov" in content_type or "quicktime" in content_type:
        ext = ".mov"
    elif "mkv" in content_type:
        ext = ".mkv"
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = Path(path).suffix or ".mp4"

    # Save to temp file (stream for larger files) with size checking
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Video at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    file_size = Path(temp_file.name).stat().st_size
    logger.info(
        f"Video downloaded: {temp_file.name} ({file_size / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_video(
    base64_string: str, max_length: int = MAX_BASE64_VIDEO_LENGTH
) -> str:
    """
    Decode base64 video to temp file and return path.

    Supports format: data:video/mp4;base64,AAAA...

    Args:
        base64_string: Base64-encoded video with data URL prefix
        max_length: Maximum allowed length of base64 string

    Returns:
        Local file path to decoded video

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 video data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Extract format and data
    if base64_string.startswith("data:video/"):
        # Format: data:video/mp4;base64,AAAA...
        header, data = base64_string.split(",", 1)
        # Extract extension from header (e.g., "data:video/mp4;base64" -> "mp4")
        format_part = header.split(";")[0]  # "data:video/mp4"
        ext = "." + format_part.split("/")[-1]  # ".mp4"
    else:
        # Assume mp4 if no header
        data = base64_string
        ext = ".mp4"

    # Decode, save, and register for cleanup
    video_bytes = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(video_bytes)
    temp_file.close()

    logger.info(
        f"Base64 video decoded: {temp_file.name} ({len(video_bytes) / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_audio(
    base64_string: str,
    *,
    default_format: str = "wav",
    max_length: int = MAX_BASE64_AUDIO_LENGTH,
) -> str:
    """Decode base64 audio to a temp file and return its path."""
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 audio data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )
    ext = "." + (default_format or "wav").lower().lstrip(".")
    if base64_string.startswith("data:audio/"):
        header, data = base64_string.split(",", 1)
        format_part = header.split(";", 1)[0]
        fmt = format_part.split("/", 1)[-1]
        if fmt:
            ext = "." + fmt.lower().replace("x-wav", "wav")
    else:
        data = base64_string
    audio_bytes = base64.b64decode(data)
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise FileSizeExceededError(
            f"Audio data exceeds maximum size: {len(audio_bytes) / 1024 / 1024:.1f} MB > "
            f"{MAX_AUDIO_SIZE / 1024 / 1024:.1f} MB limit"
        )
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    logger.info(
        f"Base64 audio decoded: {temp_file.name} ({len(audio_bytes) / 1024 / 1024:.1f} MB)"
    )
    return _temp_manager.register(temp_file.name)


def process_audio_input(audio: str | dict) -> str:
    """Process audio input in OpenAI/data-url/path/URL formats."""
    if isinstance(audio, dict):
        src = audio.get("input_audio") or audio.get("audio") or audio.get("audio_url") or audio
        if isinstance(src, dict):
            if src.get("data"):
                return decode_base64_audio(
                    str(src["data"]),
                    default_format=str(src.get("format") or "wav"),
                )
            audio = src.get("url") or src.get("path") or ""
        else:
            audio = str(src or "")
    audio = str(audio or "")
    if not audio:
        raise ValueError("Empty audio input")
    if is_base64_audio(audio):
        return decode_base64_audio(audio)
    return audio


def process_video_input(video: str | dict) -> str:
    """
    Process video input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string (data:video/mp4;base64,...)
    - OpenAI format dict: {"url": "..."} or {"url": "data:video/...;base64,..."}

    Args:
        video: Video input in any supported format

    Returns:
        Local file path to video
    """
    # Handle dict format (OpenAI style)
    if isinstance(video, dict):
        url = video.get("url", video.get("video_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        video = url

    if not video:
        raise ValueError("Empty video input")

    # Check URL/data-URL formats before filesystem probing. Long data URLs can
    # exceed platform filename limits if passed through Path.exists().
    if is_base64_video(video):
        return decode_base64_video(video)

    if is_url(video):
        return download_video(video)

    # Check if it's a local file
    try:
        if Path(video).exists():
            return video
    except OSError:
        pass

    raise ValueError(f"Cannot process video: {video[:50]}...")


def _normalize_tool_calls_for_template(tool_calls: Any) -> list[dict] | None:
    """Return OpenAI tool calls as plain mappings with dict arguments."""
    if not tool_calls:
        return None
    normalized: list[dict] = []
    for tool_call in tool_calls:
        if hasattr(tool_call, "model_dump"):
            call = tool_call.model_dump(exclude_none=True)
        elif hasattr(tool_call, "dict"):
            call = tool_call.dict()
            call = {k: v for k, v in call.items() if v is not None}
        elif isinstance(tool_call, dict):
            call = dict(tool_call)
        else:
            continue

        fn = call.get("function")
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump(exclude_none=True)
        elif hasattr(fn, "dict"):
            fn = fn.dict()
            fn = {k: v for k, v in fn.items() if v is not None}
        elif isinstance(fn, dict):
            fn = dict(fn)
        else:
            fn = {}

        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                fn["arguments"] = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        elif args is None:
            fn["arguments"] = {}

        call["function"] = fn
        normalized.append(call)

    return normalized or None


# Cache for base64 images to avoid re-saving the same image
# OrderedDict for LRU eviction with bounded size
from collections import OrderedDict

_base64_image_cache: OrderedDict[str, str] = OrderedDict()  # hash -> temp file path
_BASE64_IMAGE_CACHE_MAX_SIZE = 100


def save_base64_image(base64_string: str) -> str:
    """Save base64 image to temp file and return path. Caches identical images."""
    import hashlib

    # Hash the full base64 string to avoid collisions
    image_hash = hashlib.sha256(base64_string.encode()).hexdigest()

    # Return cached path if available and file still exists
    if image_hash in _base64_image_cache:
        cached_path = _base64_image_cache[image_hash]
        if Path(cached_path).exists():
            _base64_image_cache.move_to_end(image_hash)  # LRU: mark as recently used
            return cached_path
        else:
            # File was cleaned up, remove stale entry
            del _base64_image_cache[image_hash]

    image_bytes = decode_base64_image(base64_string)

    # Detect format from magic bytes
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif image_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        ext = ".gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        ext = ".webp"
    else:
        ext = ".jpg"  # Default

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    path = _temp_manager.register(temp_file.name)
    _base64_image_cache[image_hash] = path

    # Evict oldest entries if cache exceeds size limit
    # M5: Also delete the temp file on eviction (not just the cache entry)
    while len(_base64_image_cache) > _BASE64_IMAGE_CACHE_MAX_SIZE:
        _, evicted_path = _base64_image_cache.popitem(last=False)
        _temp_manager.cleanup(evicted_path)

    return path


def process_image_input(image: str | dict) -> str:
    """
    Process image input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string
    - OpenAI format dict: {"url": "..."} or {"url": "data:image/...;base64,..."}
    """
    # Handle dict format (OpenAI style)
    if isinstance(image, dict):
        url = image.get("url", image.get("image_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        image = url

    if not image:
        raise ValueError("Empty image input")

    # Check if it's base64 FIRST (before Path.exists() which fails on long strings)
    if is_base64_image(image):
        return save_base64_image(image)

    # Check if it's a URL
    if is_url(image):
        return download_image(image)

    # Check if it's a local file (only for short strings that could be paths)
    if len(image) < 4096 and Path(image).exists():
        return image

    raise ValueError(f"Cannot process image: {image[:50]}...")


def round_by_factor(x: int, factor: int) -> int:
    """Round to nearest multiple of factor."""
    return round(x / factor) * factor


def ceil_by_factor(x: float, factor: int) -> int:
    """Ceiling to next multiple of factor."""
    return math.ceil(x / factor) * factor


def floor_by_factor(x: float, factor: int) -> int:
    """Floor to previous multiple of factor."""
    return math.floor(x / factor) * factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: float = DEFAULT_FPS,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_FRAMES,
) -> int:
    """
    Calculate optimal number of frames to extract from video.

    Uses smart sampling based on video length and target FPS.
    """
    # Calculate duration-based frame count
    duration = total_frames / video_fps if video_fps > 0 else 0
    nframes = duration * target_fps

    # Clamp to min/max
    nframes = max(min_frames, min(nframes, max_frames, total_frames))

    # Round to factor
    nframes = max(FRAME_FACTOR, floor_by_factor(nframes, FRAME_FACTOR))

    return int(nframes)


def extract_video_frames_smart(
    video_path: str,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """
    Extract frames from video with smart sampling.

    Args:
        video_path: Path to video file
        fps: Target frames per second (default: 2.0)
        max_frames: Maximum frames to extract
        resize: Optional (width, height) to resize frames

    Returns:
        List of frame arrays (RGB format)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculate number of frames to extract
    nframes = smart_nframes(
        total_frames=total_frames,
        video_fps=video_fps,
        target_fps=fps,
        max_frames=max_frames,
    )

    # Calculate frame indices (evenly spaced)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    logger.info(
        f"Video: {total_frames} total frames @ {video_fps:.1f} fps, "
        f"extracting {nframes} frames"
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if specified
        if resize:
            frame = cv2.resize(frame, resize)

        frames.append(frame)

    cap.release()

    return frames


def save_frames_to_temp(frames: list[np.ndarray]) -> list[str]:
    """Save frame arrays to temporary files and return paths."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for frame processing")

    paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file.name, "JPEG", quality=85)
        paths.append(_temp_manager.register(temp_file.name))

    return paths


def _expand_video_placeholders_to_image_frames(
    chat_messages: list[dict],
    frame_counts: list[int],
) -> list[dict]:
    """Replace each video placeholder with N image placeholders.

    Some VLMs route videos through sampled image frames in the SimpleEngine
    path. Their prompts must contain image placeholders matching those sampled
    frames; otherwise the model sees no attached media or receives a native
    video marker unsupported by its image-only template.
    """
    if not frame_counts:
        return chat_messages

    counts = iter(frame_counts)
    rewritten: list[dict] = []
    changed = False
    for msg in chat_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            rewritten.append(msg)
            continue
        new_content: list[dict] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "video":
                count = max(1, int(next(counts, 1) or 1))
                new_content.extend({"type": "image"} for _ in range(count))
                changed = True
            else:
                new_content.append(part)
        if changed:
            out = dict(msg)
            out["content"] = new_content
            rewritten.append(out)
        else:
            rewritten.append(msg)
    return rewritten if changed else chat_messages


def _mimo_v2_token_text(processor, token_id, fallback: str) -> str:
    try:
        tokenizer = getattr(processor, "tokenizer", processor)
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert) and token_id is not None:
            token = convert(int(token_id))
            if isinstance(token, str) and token:
                return token
    except Exception:
        pass
    return fallback


class MLXMultimodalLM:
    """
    Wrapper around mlx-vlm for multimodal inference.

    This class provides a unified interface for multimodal language models
    using Apple's MLX framework. Supports:
    - Image understanding (single and multi-image)
    - Video understanding (smart frame extraction)
    - Audio understanding (for supported models)
    - OpenAI-compatible API format

    Supported models include:
    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    - LLaVA
    - Idefics3
    - PaliGemma
    - And more via mlx-vlm

    Example:
        >>> model = MLXMultimodalLM("mlx-community/Qwen2-VL-2B-Instruct-4bit")
        >>> model.load()
        >>> output = model.generate(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        enable_cache: bool = True,
        cache_size: int = 50,
    ):
        """
        Initialize the MLX multimodal language model.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable KV cache for repeated image/video+prompt (default: True)
            cache_size: Maximum cache entries (default: 50)
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.enable_cache = enable_cache

        self.model = None
        self.processor = None
        self.config = None
        self._loaded = False

        # Initialize MLLM prefix cache manager (with vision embedding caching)
        self._cache_manager: MLLMPrefixCacheManager | None = None
        if enable_cache:
            self._cache_manager = MLLMPrefixCacheManager(max_entries=cache_size)

    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        # Resolve HF repo IDs (e.g. "Org/Model") to local cache paths so that
        # file-based checks (jang_config.json, config.json) find the files.
        from ..api.utils import resolve_to_local_path
        resolved_name = resolve_to_local_path(self.model_name)
        _register_local_mlx_vlm_runtime_if_needed(resolved_name)

        # Install mlx_vlm registry patches (gemma4 + kimi_k25) on THIS thread
        # so any module-level `mx.new_stream(...)` in mlx_vlm.generate ends
        # up bound to the loader-executor worker rather than uvicorn
        # MainThread. Otherwise scheduler.step() crashes mid-prefill with
        # `RuntimeError: There is no Stream(gpu, N) in current thread.`
        # (mlxstudio#119 follow-up, 2026-05-02).
        try:
            from .. import _install_mlx_vlm_registry_patches
            _install_mlx_vlm_registry_patches()
        except Exception:
            pass

        # Apply mlx_vlm runtime compat patches (Qwen3-VL grid_thw coercion, #69).
        try:
            from ..utils.mlx_vlm_compat import apply as _apply_mlx_vlm_compat
            _apply_mlx_vlm_compat()
        except Exception as _e:
            logger.debug(f"mlx_vlm compat patch skipped: {_e}")

        # JANG VL models: use JANG loader (handles mixed-precision + mlx-vlm sanitization)
        from ..utils.jang_loader import is_jang_model
        if is_jang_model(resolved_name):
            logger.info(f"Loading JANG VL model: {self.model_name}")
            from ..utils.jang_loader import load_jang_vlm_model
            from mlx_vlm.utils import load_config
            try:
                from .. import server as _server_module
                _smelt = getattr(_server_module, '_smelt_enabled', False)
                _smelt_pct = getattr(_server_module, '_smelt_experts', 50)
            except Exception:
                _smelt = False
                _smelt_pct = 50
            if _smelt:
                from ..utils.smelt_loader import smelt_load
                self.model, self.processor = smelt_load(resolved_name, expert_percent=_smelt_pct)
            else:
                self.model, self.processor = load_jang_vlm_model(resolved_name)
            try:
                _apply_mlx_vlm_compat()
            except Exception as _e:
                logger.debug(f"mlx_vlm compat patch after JANG load skipped: {_e}")
            self.config = load_config(resolved_name)
            _set_vlm_inference_mode(self.model)
            self._loaded = True
            logger.info(f"JANG VL model loaded: {self.model_name}")
            return

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            logger.info(f"Loading MLLM: {self.model_name}")

            # Bundled Python doesn't include torch/torchvision (~2GB). Preemptively
            # patch video processor loading and suppress warnings so models with
            # video_processor sub-components (Qwen3.5-VL, InternVL, etc.) load cleanly.
            self._patch_video_processor()
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchvision.*", category=UserWarning)
                self.model, self.processor = load(self.model_name)
            if _is_mimo_v2_runtime_object_or_name(self.model, resolved_name):
                try:
                    from ..utils.jang_loader import (
                        _bind_mimo_v2_preserved_media_weights_from_index,
                    )

                    _bind_mimo_v2_preserved_media_weights_from_index(
                        self.model, Path(resolved_name)
                    )
                except Exception as _mimo_media_err:
                    raise RuntimeError(
                        "MiMo-V2 media-enabled load could not bind preserved "
                        f"visual/audio tensors: {_mimo_media_err}"
                    ) from _mimo_media_err
            self.config = load_config(self.model_name)
            if str((self.config or {}).get("model_type", "")).lower() == "zaya1_vl":
                from ..utils.jang_loader import _build_vlm_processor

                self.processor = _build_vlm_processor(Path(resolved_name))

            # vmlx#80 follow-up (Flor1an-B, 2026-04-15): mlx_vlm.load() is a
            # DIFFERENT code path from utils.tokenizer.load_model_with_fallback,
            # so neither the monkey-patch of mlx_lm.tokenizer_utils.load nor
            # the wrapper-level _inject_chat_template_if_missing ever fire for
            # standard mlx-community MLLM quants like gemma-4-31b-8bit. These
            # models ship the chat template as a sidecar chat_template.jinja
            # that HF AutoTokenizer silently ignores, and the engine crashes
            # the first time apply_chat_template is called inside batched.py.
            # Fix: inject the sidecar template directly into the processor
            # (and its inner tokenizer) right after load, before the engine
            # is marked loaded. Idempotent — bails out if a template is
            # already present.
            try:
                from ..utils.tokenizer import _inject_chat_template_if_missing as _inj
                _injected = _inj(self.processor, self.model_name)
                if _injected:
                    logger.info(
                        f"MLLM chat template injected from {_injected} for {self.model_name}"
                    )
            except Exception as _e:
                logger.debug(f"MLLM chat_template injection skipped: {_e}")

            # TurboQuant: auto-enable for ALL MLX VLM models
            _lang = getattr(self.model, 'language_model', None)
            if _lang is not None and hasattr(_lang, 'layers'):
                if _is_mimo_v2_runtime_object_or_name(self.model, resolved_name):
                    logger.info(
                        "TurboQuant skipped for MiMo-V2 MLLM load: native asymmetric SWA cache required"
                    )
                else:
                    from ..utils.tokenizer import _apply_turboquant_to_model
                    _apply_turboquant_to_model(_lang, resolved_name)

            _set_vlm_inference_mode(self.model)
            self._loaded = True
            logger.info(f"MLLM loaded successfully: {self.model_name}")

        except ImportError as e:
            # 2026-05-03: surface the ACTUAL missing module instead of
            # always blaming mlx-vlm. Common upstream causes:
            #   - torch / torchvision missing → Qwen3VLVideoProcessor and
            #     similar transformers VLM processors hard-require them.
            #     Earlier vMLX builds stripped torch from bundled-python
            #     (see panel/scripts/bundle-python.sh:128) — that strip was
            #     reversed 2026-05-03 because every VL bundle hits this.
            #   - mlx_vlm itself absent → bare wheel issue.
            #   - One of mlx_vlm's submodules (e.g. video_processor backend)
            #     transitively imports a missing dep.
            # Pre-fix we always re-raised "mlx-vlm is required" which hid
            # the real cause. Pass the original exception through.
            _msg = str(e).lower()
            if ("torch" in _msg and ("vision" in _msg or "audio" in _msg)) or "torchvision" in _msg:
                raise ImportError(
                    f"VLM bundle requires torch+torchvision in vMLX bundled-python. "
                    f"Install: <bundled-python>/bin/pip install torch torchvision. "
                    f"Original error: {e}"
                ) from e
            if "mlx_vlm" in _msg or "mlx-vlm" in _msg:
                raise ImportError(
                    f"mlx-vlm is required for multimodal inference. "
                    f"Install with: pip install mlx-vlm. "
                    f"Original error: {e}"
                ) from e
            # Unknown ImportError: re-raise WITH original cause attached so
            # the traceback shows the real failure, not a generic message.
            raise ImportError(
                f"VLM model load failed with ImportError (cause not auto-detected): {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load MLLM: {e}")
            raise

    _video_processor_patched = False

    @classmethod
    def _patch_video_processor(cls):
        """Patch AutoVideoProcessor to skip torchvision requirement.

        Some VLM models (Qwen3.5-VL, InternVL) include a video_processor
        sub-component that requires torchvision. Since we only do image inference
        and torchvision requires torch (~2GB), we patch the processor to load
        without it. Called preemptively before first model load. Idempotent.
        """
        if cls._video_processor_patched:
            return

        try:
            from transformers.models.auto import video_processing_auto

            _orig_from_pretrained = video_processing_auto.AutoVideoProcessor.from_pretrained

            @classmethod  # type: ignore[misc]
            def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                try:
                    return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
                except (ValueError, ImportError) as e:
                    if "torchvision" in str(e):
                        logger.info("Video processor skipped (torchvision not available) — image-only mode")
                        return None
                    raise

            video_processing_auto.AutoVideoProcessor.from_pretrained = _patched_from_pretrained
            cls._video_processor_patched = True
            logger.debug("Patched AutoVideoProcessor for torchvision-free loading")
        except Exception:
            logger.warning("Failed to patch AutoVideoProcessor — torchvision-dependent VLMs may fail to load")
            cls._video_processor_patched = True  # Don't retry on every load, log warning once

    def _prepare_images(self, images: list) -> list[str]:
        """Process image inputs and return local file paths."""
        processed = []
        failed_count = 0
        for img in images:
            try:
                path = process_image_input(img)
                processed.append(path)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to process image: {e}")
        if images and not processed:
            raise ValueError(
                f"All {failed_count} image(s) failed to process. "
                f"Check image URLs/paths and try again."
            )
        return processed

    def _prepare_audio(self, audio_inputs: list) -> list[str]:
        """Process audio inputs and return local paths/URLs loadable by mlx-vlm."""
        processed = []
        failed_count = 0
        for audio in audio_inputs:
            try:
                path = process_audio_input(audio)
                if path:
                    processed.append(path)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to process audio: {e}")
        if audio_inputs and not processed:
            raise ValueError(
                f"All {failed_count} audio input(s) failed to process. "
                f"Check audio URLs/paths and try again."
            )
        return processed

    def _prepare_video(
        self,
        video_input: str | dict,
        fps: float = DEFAULT_FPS,
        max_frames: int = MAX_FRAMES,
    ) -> list[str]:
        """
        Process video input and extract frames.

        Supports:
        - Local file paths
        - URLs (http/https) - will be downloaded
        - Base64 encoded videos (data:video/mp4;base64,...)
        - OpenAI format dicts: {"url": "..."} or {"video_url": {"url": "..."}}

        Args:
            video_input: Video in any supported format
            fps: Frames per second to extract
            max_frames: Maximum frames to extract

        Returns:
            List of paths to extracted frame images
        """
        # Process video input (download if URL, decode if base64)
        video_path = process_video_input(video_input)

        # Extract frames
        frames = extract_video_frames_smart(
            video_path,
            fps=fps,
            max_frames=max_frames,
        )
        return save_frames_to_temp(frames)

    def _guard_simple_image_prefill(
        self,
        formatted_prompt: str,
        has_images: bool,
        *,
        images: list | None = None,
        resize_shape: object | None = None,
    ) -> None:
        """Apply the vmlx#156 image-prefill guard on SimpleEngine paths.

        Continuous-batching VLMs run this guard inside
        ``MLLMBatchGenerator._run_vision_encoding_inner``. Direct
        ``MLXMultimodalLM`` generate/chat paths still call ``mlx_vlm``
        one-shot APIs, so they need the same preflight before Metal sees an
        image-expanded prompt. Use processor-expanded ``input_ids`` when
        possible because each VLM family can expand one image/video placeholder
        into a different number of language tokens.
        """
        if not has_images:
            return
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            mx = None  # type: ignore[assignment]

        try:
            seq_len = self._simple_vlm_prefill_seq_len(
                formatted_prompt,
                images=images,
                resize_shape=resize_shape,
            )
        except Exception as exc:
            logger.debug("Simple VLM image-prefill guard skipped: tokenize failed: %s", exc)
            return

        try:
            if mx is not None:
                mx.clear_cache()
        except Exception:
            pass

        from vmlx_engine.mllm_batch_generator import (
            _raise_if_image_prefill_exceeds_budget,
        )

        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            inner = getattr(self.model, "model", None)
            language_model = getattr(inner, "language_model", None)
        _raise_if_image_prefill_exceeds_budget(
            has_images=True,
            seq_len=seq_len,
            language_model=language_model,
        )

    def _simple_vlm_prefill_seq_len(
        self,
        formatted_prompt: str,
        *,
        images: list | None = None,
        resize_shape: object | None = None,
    ) -> int:
        """Return the best available VLM prefill length for direct paths."""

        def _seq_len_from_input_ids(input_ids: object) -> int | None:
            if input_ids is None:
                return None
            shape = getattr(input_ids, "shape", None)
            if shape:
                if len(shape) >= 2:
                    return int(shape[1])
                return int(shape[0])
            if isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    return int(len(input_ids[0]))
                return int(len(input_ids))
            return None

        if images:
            try:
                from mlx_vlm.generate import normalize_resize_shape, prepare_inputs

                model_config = getattr(self.model, "config", None)
                image_token_index = (
                    getattr(model_config, "image_token_index", None)
                    if model_config is not None else None
                )
                if image_token_index is None and model_config is not None:
                    image_token_index = getattr(model_config, "image_token_id", None)
                model_type = str(getattr(model_config, "model_type", "") or "")
                add_special_tokens = (
                    getattr(self.processor, "chat_template", None) is None
                    if model_type in {"gemma3", "gemma3n", "gemma4"}
                    else True
                )
                inputs = prepare_inputs(
                    self.processor,
                    images=images,
                    prompts=formatted_prompt,
                    image_token_index=image_token_index,
                    resize_shape=normalize_resize_shape(resize_shape),
                    add_special_tokens=add_special_tokens,
                )
                seq_len = _seq_len_from_input_ids(inputs.get("input_ids"))
                if seq_len:
                    return seq_len
            except Exception as exc:
                logger.debug(
                    "Simple VLM image-prefill guard using raw tokenizer length; "
                    "processor expansion probe failed: %s",
                    exc,
                )

        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        return len(tokenizer.encode(formatted_prompt))

    @staticmethod
    def _extract_multimodal_messages(
        messages: list[dict],
    ) -> tuple[list[dict], list[str], list, list]:
        """
        Parse OpenAI-format messages into chat_messages, image URLs, videos, and audio.

        Extracts text, images, and videos from multimodal message content,
        building properly structured chat messages for Qwen3-VL-MoE and similar
        models that expect image tokens before text in user messages.

        Args:
            messages: List of chat messages in OpenAI format

        Returns:
            Tuple of (chat_messages, all_image_urls, videos, audio_inputs) where:
            - chat_messages: Properly structured messages for chat template
            - all_image_urls: Raw image URLs/paths to process
            - videos: Raw video inputs to process
            - audio_inputs: Raw audio inputs to process
        """
        all_image_urls: list[str] = []
        videos: list = []
        audio_inputs: list = []
        chat_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_text = ""
            msg_image_count = 0
            msg_video_count = 0
            msg_audio_count = 0

            if isinstance(content, str):
                msg_text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        msg_text += item
                        continue

                    # Convert Pydantic models to dicts.
                    # exclude_none=True prevents Jinja2 false key-existence bugs
                    # (checking 'if image_url' returns True for None values). If a
                    # field should be explicitly None (not absent), it would need
                    # special handling.
                    if hasattr(item, "model_dump"):
                        item = item.model_dump(exclude_none=True)
                    elif hasattr(item, "dict"):
                        item = item.dict()
                        # Remove None values to match exclude_none behavior
                        item = {k: v for k, v in item.items() if v is not None}

                    if isinstance(item, dict):
                        item_type = item.get("type", "")

                        if item_type in ("text", "input_text"):
                            msg_text += item.get("text", "")

                        elif item_type == "image_url":
                            img_url = item.get("image_url", {})
                            if isinstance(img_url, str):
                                all_image_urls.append(img_url)
                            else:
                                all_image_urls.append(img_url.get("url", ""))
                            msg_image_count += 1

                        elif item_type == "input_image":
                            img_url = item.get("image_url", item.get("url", ""))
                            if isinstance(img_url, str):
                                all_image_urls.append(img_url)
                            elif isinstance(img_url, dict):
                                all_image_urls.append(img_url.get("url", ""))
                            msg_image_count += 1

                        elif item_type == "image":
                            all_image_urls.append(
                                item.get("image", item.get("url", ""))
                            )
                            msg_image_count += 1

                        elif item_type == "video":
                            videos.append(item.get("video", item.get("url", "")))
                            msg_video_count += 1

                        elif item_type == "video_url":
                            vid_url = item.get("video_url", {})
                            if isinstance(vid_url, str):
                                videos.append(vid_url)
                            else:
                                videos.append(vid_url.get("url", ""))
                            msg_video_count += 1

                        elif item_type == "input_video":
                            vid_url = item.get("video_url", item.get("url", item.get("file_id", "")))
                            if isinstance(vid_url, str):
                                videos.append(vid_url)
                            else:
                                videos.append(vid_url.get("url", ""))
                            msg_video_count += 1

                        elif item_type in ("input_audio", "audio", "audio_url"):
                            src = item.get("input_audio") or item.get("audio") or item.get("audio_url") or item
                            audio_inputs.append(src)
                            msg_audio_count += 1

            # Build properly structured message for Qwen3-VL-MoE
            # Format: {"role": "...", "content": [{"type": "image"|"video"}, ..., {"type": "text", "text": "..."}]}
            # Preserve tool_calls on assistant messages and tool role fields
            tool_calls = _normalize_tool_calls_for_template(msg.get("tool_calls"))
            tool_call_id = msg.get("tool_call_id")
            msg_name = msg.get("name")

            if msg_text or msg_image_count > 0 or msg_video_count > 0 or msg_audio_count > 0 or tool_calls or role == "tool":
                if (msg_image_count > 0 or msg_video_count > 0 or msg_audio_count > 0) and role in ("user", "assistant"):
                    # Build multimodal content list with image markers for
                    # any role that carries media (user or assistant).
                    # Some UIs (e.g. klite) send images on assistant messages
                    # when attaching an image to the conversation context.
                    content_list: list[dict] = []
                    if msg_audio_count and not msg_image_count and not msg_video_count:
                        content_list.append(
                            {"type": "text", "text": msg_text, "content": msg_text}
                        )
                        for _ in range(msg_audio_count):
                            content_list.append({"type": "audio"})
                    else:
                        for _ in range(msg_image_count):
                            content_list.append({"type": "image"})
                        for _ in range(msg_video_count):
                            content_list.append({"type": "video"})
                        for _ in range(msg_audio_count):
                            content_list.append({"type": "audio"})
                        content_list.append(
                            {"type": "text", "text": msg_text, "content": msg_text}
                        )
                    out_msg: dict = {"role": role, "content": content_list}
                    if role == "assistant" and tool_calls:
                        out_msg["tool_calls"] = tool_calls
                    chat_messages.append(out_msg)
                elif role == "assistant":
                    out_msg = {"role": role, "content": msg_text}
                    if tool_calls:
                        out_msg["tool_calls"] = tool_calls
                    chat_messages.append(out_msg)
                elif role == "tool":
                    out_msg = {"role": role, "content": msg_text}
                    if tool_call_id:
                        out_msg["tool_call_id"] = tool_call_id
                    if msg_name:
                        out_msg["name"] = msg_name
                    chat_messages.append(out_msg)
                else:
                    chat_messages.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "text", "text": msg_text, "content": msg_text}
                            ],
                        }
                    )

        return chat_messages, all_image_urls, videos, audio_inputs

    def _normalize_text_only_messages_for_processor(
        self,
        chat_messages: list[dict],
        *,
        has_media: bool,
    ) -> list[dict]:
        """Use family-native plain text turns when a VLM request has no media.

        Some local processor templates have two distinct modes: rich list
        content is required for real image/video turns, while text-only turns
        render correctly as plain strings. Feeding a no-media request through
        the generic rich-content MLLM shape can make the template path look
        like a multimodal prompt and weakens exact text, multi-turn, and
        reasoning probes. Keep media turns untouched; only collapse text-only
        content lists that contain text fragments and no modality markers.
        """

        if has_media:
            return chat_messages

        config = getattr(self, "config", None)
        if isinstance(config, dict):
            model_type = str(config.get("model_type", "") or "").lower()
        else:
            model_type = str(getattr(config, "model_type", "") or "").lower()
        if model_type not in {"zaya1_vl", "mimo_v2"}:
            return chat_messages

        normalized: list[dict] = []
        for message in chat_messages:
            content = message.get("content")
            if not isinstance(content, list):
                normalized.append(message)
                continue

            text_parts: list[str] = []
            contains_media_marker = False
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if not isinstance(item, dict):
                    text_parts.append(str(item))
                    continue
                item_type = str(item.get("type", "") or "").lower()
                if item_type and item_type != "text":
                    contains_media_marker = True
                    break
                text_parts.append(str(item.get("text") or item.get("content") or ""))

            if contains_media_marker:
                normalized.append(message)
                continue

            collapsed = dict(message)
            collapsed["content"] = "".join(text_parts)
            normalized.append(collapsed)

        if model_type == "mimo_v2":
            leading_system_parts: list[str] = []
            folded: list[dict] = []
            folded_into_user = False
            for message in normalized:
                role = str(message.get("role") or "").lower()
                if not folded and role in {"system", "developer"}:
                    text = str(message.get("content") or "").strip()
                    if text:
                        leading_system_parts.append(text)
                    continue
                if (
                    leading_system_parts
                    and not folded_into_user
                    and role == "user"
                ):
                    combined = dict(message)
                    user_text = str(combined.get("content") or "").strip()
                    combined["content"] = "\n\n".join(
                        part for part in [*leading_system_parts, user_text] if part
                    )
                    folded.append(combined)
                    folded_into_user = True
                    continue
                folded.append(message)
            if leading_system_parts and not folded_into_user:
                folded.insert(
                    0,
                    {
                        "role": "user",
                        "content": "\n\n".join(leading_system_parts),
                    },
                )
            normalized = folded

        return normalized

    def _normalize_mimo_audio_messages_for_template(
        self,
        chat_messages: list[dict],
    ) -> list[dict]:
        """Render MiMo audio content parts as explicit artifact audio tokens.

        The current local MiMo bundles use a Qwen2.5-VL processor shim. That
        processor preserves image/video content parts, but ignores raw
        ``{"type": "audio"}`` parts when rendering the chat template. The
        audio code bridge can only splice embeddings if the prompt contains
        the model-owned audio pad token, so normalize MiMo audio placeholders
        to the special token sequence declared in ``processor_config``.
        """

        config = getattr(self, "config", None)
        if isinstance(config, dict):
            model_type = str(config.get("model_type", "") or "").lower()
            processor_config = config.get("processor_config") or {}
        else:
            model_type = str(getattr(config, "model_type", "") or "").lower()
            processor_config = getattr(config, "processor_config", None) or {}
        if model_type != "mimo_v2" or not isinstance(processor_config, dict):
            return chat_messages
        audio_token_id = processor_config.get("audio_token_id")
        if audio_token_id is None:
            return chat_messages
        audio_marker = _mimo_v2_token_text(
            self.processor,
            processor_config.get("audio_start_token_id"),
            "<|mimo_audio_start|>",
        )
        audio_marker += _mimo_v2_token_text(
            self.processor,
            audio_token_id,
            "<|audio_pad|>",
        )
        audio_marker += _mimo_v2_token_text(
            self.processor,
            processor_config.get("audio_end_token_id"),
            "<|mimo_audio_end|>",
        )

        rewritten: list[dict] = []
        changed = False
        for message in chat_messages:
            content = message.get("content")
            if not isinstance(content, list):
                rewritten.append(message)
                continue
            new_content: list[dict] = []
            for part in content:
                if isinstance(part, dict) and str(part.get("type") or "").lower() in {
                    "audio",
                    "input_audio",
                    "audio_url",
                }:
                    new_content.append({"type": "text", "text": audio_marker, "content": audio_marker})
                    changed = True
                else:
                    new_content.append(part)
            if changed:
                out = dict(message)
                out["content"] = new_content
                rewritten.append(out)
            else:
                rewritten.append(message)
        return rewritten if changed else chat_messages

    def _synthesizes_thinking_prompt_when_enabled(self) -> bool:
        """Return True for MLLM families whose template needs an explicit open rail.

        Some MLLM bundles declare qwen3 reasoning support while their mlx-vlm
        chat template can be a plain ``assistant: `` suffix. In that case an
        explicit per-chat/API ``enable_thinking=true`` may need to open the
        qwen3 rail in the prompt. This is intentionally family-gated; most VLM
        templates either own their think rail or do not support reasoning. The
        current ZAYA1-VL plain-template artifacts are registry-demoted to
        ``supports_thinking=False`` because live proof produced hidden-only
        output under the synthetic rail.
        """

        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            config = {}

        caps = config.get("capabilities")
        caps = caps if isinstance(caps, dict) else {}
        if caps.get("supports_thinking") is False:
            return False

        model_type = config.get("model_type")
        text_config = config.get("text_config")
        if not model_type and isinstance(text_config, dict):
            model_type = text_config.get("model_type")

        family_names = {
            str(v).lower()
            for v in (caps.get("family"), model_type)
            if isinstance(v, str) and v
        }
        reasoning_parser = caps.get("reasoning_parser")
        supports_thinking = caps.get("supports_thinking")
        think_in_template = caps.get("think_in_template")

        try:
            from vmlx_engine.model_config_registry import get_model_config_registry

            registry = get_model_config_registry()
            family_config = registry.lookup(str(getattr(self, "model_name", "") or ""))
            if getattr(family_config, "family_name", None):
                family_names.add(str(family_config.family_name).lower())
            if getattr(family_config, "reasoning_parser", None):
                reasoning_parser = family_config.reasoning_parser
            if getattr(family_config, "supports_thinking", None) is not None:
                supports_thinking = family_config.supports_thinking
            if getattr(family_config, "think_in_template", None) is not None:
                think_in_template = family_config.think_in_template
        except Exception:
            pass

        if supports_thinking is False:
            return False
        if think_in_template is True:
            return False
        if str(reasoning_parser or "").lower() != "qwen3":
            return False
        return bool(family_names & {"zaya", "zaya1_vl", "zaya1-vl"})

    def _prompt_template_supports_thinking(self) -> bool:
        """Return whether the native VLM prompt template owns the think rail.

        Capability stamps are the source of truth here. A model can support a
        qwen-style reasoning parser while its VLM chat template remains plain;
        those families must not receive template-sentinel assumptions when
        ``enable_thinking=false``.
        """

        config = getattr(self, "config", None)
        caps = config.get("capabilities") if isinstance(config, dict) else None
        caps = caps if isinstance(caps, dict) else {}
        if caps.get("supports_thinking") is False:
            return False
        if caps.get("think_in_template") is not None:
            return bool(caps.get("think_in_template"))

        try:
            from vmlx_engine.model_config_registry import get_model_config_registry

            family_config = get_model_config_registry().lookup(
                str(getattr(self, "model_name", "") or "")
            )
            if getattr(family_config, "supports_thinking", None) is False:
                return False
            if getattr(family_config, "think_in_template", None) is not None:
                return bool(family_config.think_in_template)
        except Exception:
            pass

        return False

    def _apply_chat_template(
        self,
        chat_messages: list[dict],
        enable_thinking: bool | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """
        Apply chat template to structured messages with enable_thinking support.

        Handles TypeError fallback for processors that don't support
        enable_thinking and general exception fallback to last user message.
        The wrapper preserves the native template output; family-specific
        no-thinking prompt sentinels must live in explicit renderer contracts.

        Args:
            chat_messages: Structured chat messages from _extract_multimodal_messages()
            enable_thinking: Whether to enable thinking mode (None = don't pass)

        Returns:
            Formatted prompt string
        """
        from mlx_vlm.prompt_utils import get_chat_template

        template_kwargs = {}
        model_type = ""
        try:
            config = getattr(self, "config", None)
            if isinstance(config, dict):
                model_type = str(config.get("model_type", "") or "").lower()
            else:
                model_type = str(getattr(config, "model_type", "") or "").lower()
        except Exception:
            model_type = ""
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if tools:
            template_kwargs["tools"] = tools
        if model_type == "mimo_v2":
            chat_messages = self._normalize_mimo_audio_messages_for_template(chat_messages)

        formatted_prompt = None
        try:
            formatted_prompt = get_chat_template(
                self.processor,
                chat_messages,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError:
            # Processor doesn't support enable_thinking kwarg — use processor
            # WITHOUT the kwarg first. Preserve tools unless the processor
            # rejects them too.
            fallback_kwargs = dict(template_kwargs)
            fallback_kwargs.pop("enable_thinking", None)
            try:
                formatted_prompt = get_chat_template(
                    self.processor,
                    chat_messages,
                    add_generation_prompt=True,
                    **fallback_kwargs,
                )
            except TypeError:
                fallback_kwargs.pop("tools", None)
                try:
                    formatted_prompt = get_chat_template(
                        self.processor,
                        chat_messages,
                        add_generation_prompt=True,
                        **fallback_kwargs,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply chat template: {e}, using last user message"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}, using last user message"
                )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, using last user message"
            )

        # Some processor templates can be plain while the model is still
        # qwen3-reasoning capable. Honor explicit thinking-on by opening the
        # rail in the assistant generation prompt only for registry-approved
        # families; Auto/Off remain plain.
        if (
            enable_thinking is True
            and formatted_prompt
            and self._synthesizes_thinking_prompt_when_enabled()
        ):
            last_think = formatted_prompt.rfind("<think>")
            has_open_unclosed = last_think >= 0 and "</think>" not in formatted_prompt[last_think:]
            if not has_open_unclosed:
                stripped = formatted_prompt.rstrip()
                if stripped.endswith("assistant:"):
                    formatted_prompt = stripped + " <think>\n"
                else:
                    formatted_prompt = stripped + "\n<think>\n"

        if enable_thinking is False and formatted_prompt:
            try:
                from ..utils.chat_template_kwargs import ensure_thinking_off_sentinel

                formatted_prompt = ensure_thinking_off_sentinel(
                    formatted_prompt,
                    family_name=str((self.config or {}).get("model_type", ""))
                    if isinstance(self.config, dict)
                    else None,
                    model_name=self.model_name,
                    tools_present=bool(tools),
                )
            except Exception as exc:
                logger.debug(
                    "MLLM thinking-off prompt sentinel skipped for %s: %s",
                    self.model_name,
                    exc,
                )

        if formatted_prompt is None:
            # Fallback to last user message if template fails
            last_user_msg = ""
            for m in reversed(chat_messages):
                if m["role"] == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                last_user_msg = item.get("text", "")
                                break
                    else:
                        last_user_msg = content
                    break
            formatted_prompt = last_user_msg

        return formatted_prompt

    def generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list | None = None,
        audio: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        use_cache: bool = True,
        **kwargs,
    ) -> MLLMOutput:
        """
        Generate text from multimodal input.

        Args:
            prompt: Text prompt/question
            images: List of image paths, URLs, or base64 strings
            videos: List of video inputs (paths, URLs, base64, or OpenAI format dicts)
            audio: List of audio file paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            video_fps: FPS for video frame extraction (default: 2.0)
            video_max_frames: Max frames to extract from video
            use_cache: Whether to use KV cache (default: True)
            **kwargs: Additional generation parameters

        Returns:
            MLLMOutput with generated text

        Example:
            # With local video
            output = model.generate("Describe this video", videos=["video.mp4"])

            # With video URL
            output = model.generate("What happens?", videos=["https://example.com/video.mp4"])

            # With base64 video
            output = model.generate("Describe", videos=["data:video/mp4;base64,AAAA..."])
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate
        from mlx_vlm.models import cache as vlm_cache
        from mlx_vlm.prompt_utils import apply_chat_template

        images = images or []
        videos = videos or []
        audio = audio or []

        # Process all images (including frames from videos) and audio.
        all_images = []
        all_audio = []
        all_sources = []  # Track original sources for cache key

        # Process image inputs
        if images:
            all_images.extend(self._prepare_images(images))
            all_sources.extend(images)
        if audio:
            all_audio.extend(self._prepare_audio(audio))
            all_sources.extend(audio)

        # Extract frames from videos
        for video_path in videos:
            frames = self._prepare_video(
                video_path,
                fps=video_fps,
                max_frames=video_max_frames,
            )
            all_images.extend(frames)
            # Include video params in cache key
            video_str = video_path if isinstance(video_path, str) else str(video_path)
            all_sources.append(
                f"video:{video_str}:fps{video_fps}:max{video_max_frames}"
            )
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template if needed
        if all_images and hasattr(self.processor, "apply_chat_template"):
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        # Check cache for existing KV state
        prompt_cache = None
        prompt_cache_state = None
        cache_hit = False
        _cache_token_ids: list[int] | None = None  # reused by store path below
        media_cache_safe = _supports_simple_media_prompt_cache(self.model)

        if use_cache and media_cache_safe and self._cache_manager is not None and all_sources:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = _media_expanded_token_ids_for_cache(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    images=all_images,
                    resize_shape=kwargs.get("resize_shape"),
                ) or tokenizer.encode(formatted_prompt)
                _cache_token_ids = token_ids
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_sources, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    if all_images:
                        prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                        if prompt_cache_state is None:
                            logger.debug(
                                "[PREFIX CACHE] Generate image hit lacked PromptCacheState; "
                                "falling back to full prompt processing"
                            )
                        else:
                            logger.debug(
                                "[PREFIX CACHE] Generate image hit - using mlx-vlm "
                                "PromptCacheState for prefix trim"
                            )
                    else:
                        prompt_cache = cache_entry.kv_cache
                    cache_hit = True
                    logger.info(
                        f"MLLM cache hit for {len(all_sources)} source(s)"
                        + (f", {prefix_match_len} prefix tokens match" if prefix_match_len > 0 else "")
                    )
            except Exception as e:
                logger.debug(f"Generate cache fetch failed: {e}")

        # Create new cache if needed
        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        # Generate with cache
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
            prompt_cache=prompt_cache,
            **kwargs,
        )

        # Store cache for future reuse (only on miss).
        # Use `is not None` — MLLMPrefixCacheManager.__len__ exists but
        # __bool__ does not, so plain `if self._cache_manager` evaluates
        # False when the cache is empty (via len fallback). That made
        # store skip the very first attempt, so the cache could never
        # populate and no request ever hit. (2026-04-18 Ralph iter 11)
        if use_cache and media_cache_safe and self._cache_manager is not None and all_sources and not cache_hit:
            if prompt_cache is not None:
                try:
                    num_tokens = getattr(result, "prompt_tokens", 0)
                    cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                        prompt_cache, num_tokens
                    )
                    # Prefer the real token_ids (computed above during fetch)
                    # so partial-hit prefix matching works. store_cache's
                    # num_tokens-only path writes dummy `[0] * N` which makes
                    # downstream get_prefix_match_length always return 0.
                    if _cache_token_ids:
                        self._cache_manager.store(
                            images=all_sources,
                            prompt=formatted_prompt,
                            vision_embeddings=None,
                            kv_cache=cache_to_store,
                            token_ids=_cache_token_ids,
                            num_image_tokens=0,
                            model_name=self.model_name,
                        )
                    else:
                        self._cache_manager.store_cache(
                            all_sources, formatted_prompt, cache_to_store, num_tokens
                        )
                    logger.info(f"MLLM cache stored for {len(all_sources)} source(s)")
                except Exception as e:
                    logger.debug(f"Failed to store MLLM cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        finish_reason = "length" if generation_tokens >= max_tokens else "stop"

        return MLLMOutput(
            text=output_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
        )

    def stream_generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream text generation for multimodal input.

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_fps: FPS for video frame extraction
            video_max_frames: Maximum video frames to extract
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
            from mlx_vlm.models import cache as vlm_cache
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError:
            # Fallback to non-streaming
            output = self.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                video_fps=video_fps,
                **kwargs,
            )
            yield output.text
            return

        images = images or []
        videos = videos or []

        # Process images
        all_images = []
        all_sources = []
        if images:
            all_images.extend(self._prepare_images(images))
            all_sources.extend(images)
        for video_path in videos:
            frames = self._prepare_video(video_path, fps=video_fps, max_frames=video_max_frames)
            all_images.extend(frames)
            video_str = video_path if isinstance(video_path, str) else str(video_path)
            all_sources.append(
                f"video:{video_str}:fps{video_fps}:max{video_max_frames}"
            )

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        if all_images:
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        prompt_cache = None
        prompt_cache_state = None
        cache_hit = False
        token_ids: list[int] = []
        use_cache = kwargs.pop("use_cache", True)
        media_cache_safe = _supports_simple_media_prompt_cache(self.model)

        if use_cache and media_cache_safe and self._cache_manager is not None and all_sources:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = _media_expanded_token_ids_for_cache(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    images=all_images,
                    resize_shape=kwargs.get("resize_shape"),
                ) or tokenizer.encode(formatted_prompt)
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_sources, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    if all_images:
                        prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                        if prompt_cache_state is None:
                            logger.debug(
                                "[PREFIX CACHE] Stream generate image hit lacked "
                                "PromptCacheState; falling back to full prompt processing"
                            )
                    else:
                        prompt_cache = cache_entry.kv_cache
                    cache_hit = True
                    if prefix_match_len > 0:
                        logger.debug(
                            f"Stream generate prefix cache hit: {prefix_match_len} tokens, "
                            f"{len(all_sources)} source(s)"
                        )
            except Exception as e:
                logger.debug(f"Stream generate cache fetch failed: {e}")

        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        last_prompt_tokens = 0
        with _MaybeVLMStream():
            for chunk in stream_generate(
                self.model,
                self.processor,
                formatted_prompt,
                all_images if all_images else None,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_cache=prompt_cache,
                **kwargs,
            ):
                chunk_prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                if chunk_prompt_tokens:
                    last_prompt_tokens = chunk_prompt_tokens
                yield chunk

        if (
            use_cache
            and media_cache_safe
            and self._cache_manager is not None
            and all_sources
            and not cache_hit
            and prompt_cache
        ):
            try:
                prompt_tokens_count = last_prompt_tokens or len(token_ids)
                cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                    prompt_cache, prompt_tokens_count
                )
                self._cache_manager.store(
                    images=all_sources,
                    prompt=formatted_prompt,
                    vision_embeddings=None,
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=0,
                    model_name=self.model_name,
                )
                logger.info(
                    f"MLLM stream cache stored for {len(all_sources)} source(s)"
                )
            except Exception as e:
                logger.debug(f"Failed to store MLLM stream cache: {e}")

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> MLLMOutput:
        """
        Chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            MLLMOutput with assistant's response
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate

        # Extract text, images, videos, and audio from messages
        chat_messages, all_image_urls, videos, audio_inputs = self._extract_multimodal_messages(messages)
        chat_messages = self._normalize_text_only_messages_for_processor(
            chat_messages,
            has_media=bool(all_image_urls or videos or audio_inputs),
        )

        logger.info(f"MLLM.chat() called with {len(messages)} messages")

        # Process images
        all_images = []
        all_audio = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))
        if audio_inputs:
            all_audio.extend(self._prepare_audio(audio_inputs))

        # Process videos
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        video_frame_counts: list[int] = []
        for video_path in videos:
            frames = self._prepare_video(
                video_path, fps=video_fps, max_frames=video_max_frames
            )
            all_images.extend(frames)
            video_frame_counts.append(len(frames))
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        model_type = (
            str(self.config.get("model_type", "") or "").lower()
            if isinstance(self.config, dict)
            else str(getattr(self.config, "model_type", "") or "").lower()
        )
        if model_type in {"gemma4_unified", "step3p7"} and video_frame_counts:
            chat_messages = _expand_video_placeholders_to_image_frames(
                chat_messages,
                video_frame_counts,
            )

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        template_tools = kwargs.pop("tools", None)
        logger.info(
            f"Applying chat template with {len(chat_messages)} messages, "
            f"{len(all_images)} images, {len(all_audio)} audio"
        )
        for i, cm in enumerate(chat_messages):
            content_preview = str(cm.get("content", ""))[:80]
            logger.info(
                f"  Chat msg {i}: role={cm['role']}, content={content_preview}..."
            )
        enable_thinking = kwargs.pop("enable_thinking", None)
        formatted_prompt = self._apply_chat_template(
            chat_messages,
            enable_thinking,
            tools=template_tools,
        )

        # Post-template image count guard: VLM chat templates may not expand
        # image placeholders for assistant-role messages. Trim all_images to
        # match actual placeholder token count to prevent crashes in generate().
        if all_images and self.config:
            _img_token_id = getattr(self.config, "image_token_index", None)
            if _img_token_id is not None:
                _tok = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                try:
                    _token_ids = _tok.encode(formatted_prompt)
                    _slot_count = _token_ids.count(_img_token_id)
                    if _slot_count != len(all_images):
                        logger.warning(
                            f"Image count mismatch: {len(all_images)} images but "
                            f"{_slot_count} slots in prompt (likely assistant-role "
                            f"images not supported by template). Trimming to {_slot_count}."
                        )
                        all_images = all_images[:_slot_count]
                except Exception as _e:
                    logger.debug(f"Image slot count check failed: {_e}")

        # Prefix caching with vision embedding support
        # Following LMCache approach: cache vision embeddings to skip encoder on hit
        from mlx_vlm.models import cache as vlm_cache
        import time

        use_cache = kwargs.pop("use_cache", True)
        cache_entry = None
        prefix_match_len = 0
        cache_hit = False

        # Tokenize prompt for cache lookup
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        token_ids = _media_expanded_token_ids_for_cache(
            self.model,
            self.processor,
            formatted_prompt,
            images=all_images,
            audio=all_audio,
            resize_shape=kwargs.get("resize_shape"),
        ) or tokenizer.encode(formatted_prompt)
        media_cache_safe = _supports_simple_media_prompt_cache(self.model)

        # Check prefix cache
        if use_cache and media_cache_safe and self._cache_manager is not None and all_images:
            try:
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry:
                    cache_hit = True
                    # NOTE: cache_entry.vision_embeddings is not used — mlx-vlm's
                    # generate() does not accept pre-computed vision embeddings.
                    # The KV cache hit path below handles the actual speedup.
                    if prefix_match_len > 0:
                        logger.debug(
                            f"[PREFIX CACHE] {prefix_match_len} prefix tokens match"
                        )
            except Exception as e:
                logger.warning(f"Cache fetch failed: {e}")

        # Generate - use KV cache if available from previous identical request
        start_time = time.time()

        # Create or reuse prompt cache for prefix caching speedup
        prompt_cache = None
        prompt_cache_state = None
        skip_prompt_processing = False

        if media_cache_safe and cache_hit and cache_entry and cache_entry.kv_cache:
            if all_images:
                # mlx-vlm trims multimodal prefixes via PromptCacheState. Passing
                # raw prompt_cache with the full prompt would double-process the
                # image/prefix tokens; dropping the hit would full-prefill every
                # follow-up turn.
                prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                if prompt_cache_state is None:
                    logger.debug(
                        "[PREFIX CACHE] Image prefix hit lacked PromptCacheState; "
                        "falling back to full prompt processing"
                    )
                else:
                    logger.debug(
                        "[PREFIX CACHE] Image prefix hit - using mlx-vlm "
                        "PromptCacheState for prefix trim"
                    )
                skip_prompt_processing = False
            else:
                # Text-only: can use skip_prompt_processing for maximum speedup
                logger.debug(
                    "[PREFIX CACHE] Text-only cache hit - using skip_prompt_processing speedup"
                )
                cached_prompt_cache = cache_entry.kv_cache
                try:
                    import copy

                    prompt_cache = []
                    for layer_cache in cached_prompt_cache:
                        new_cache = copy.copy(layer_cache)
                        if hasattr(layer_cache, "state"):
                            state = layer_cache.state
                            if state is not None:
                                import mlx.core as mx

                                if len(state) >= 2 and state[0] is not None:
                                    new_cache.keys = mx.array(state[0])
                                    new_cache.values = mx.array(state[1])
                                    if len(state) >= 3:
                                        new_cache.offset = state[2]
                                    elif hasattr(layer_cache, "offset"):
                                        new_cache.offset = layer_cache.offset
                        prompt_cache.append(new_cache)
                    skip_prompt_processing = True
                    logger.debug(
                        f"[PREFIX CACHE] Skipping {prefix_match_len} token forward pass"
                    )
                except Exception as e:
                    logger.warning(f"[PREFIX CACHE] Failed to copy cache: {e}")
                    prompt_cache = None
                    skip_prompt_processing = False

        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            # Create fresh cache
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Extract sampling params not natively supported by mlx_vlm.generate()
        # (top_k, min_p require a custom sampler; they're silently ignored otherwise)
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            from ..sampling import make_sampler
            top_p = kwargs.pop("top_p", 1.0)
            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
            prompt_cache=prompt_cache,
            skip_prompt_processing=skip_prompt_processing,
            **kwargs,
        )

        # Store KV cache for future reuse (on cache miss)
        # IMPORTANT: We need to store only the prompt portion, not generated tokens
        if (
            use_cache
            and media_cache_safe
            and self._cache_manager is not None
            and all_images
            and not cache_hit
            and prompt_cache
        ):
            try:
                # Get prompt token count (before generation)
                prompt_tokens_count = getattr(result, "prompt_tokens", 0)

                cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                    prompt_cache, prompt_tokens_count
                )

                # Estimate num_image_tokens from the model config or token IDs.
                # Different models use different counts (e.g., Gemma3=256, Qwen2-VL varies).
                num_img_tokens = 0
                if all_images and self.config:
                    # Try to get image_token_index from config and count occurrences in token_ids
                    img_token_id = getattr(self.config, "image_token_index", None)
                    if img_token_id is not None and token_ids:
                        num_img_tokens = token_ids.count(img_token_id)
                    if num_img_tokens == 0:
                        # Fallback: use prompt_tokens_count minus a rough text estimate
                        # or just leave as 0 (cache stats only, not critical for correctness)
                        num_img_tokens = 0

                self._cache_manager.store(
                    images=all_images,
                    prompt=formatted_prompt,
                    vision_embeddings=None,  # mlx-vlm doesn't support pre-computed embeddings
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=num_img_tokens,
                    model_name=self.model_name,
                )
                logger.info(
                    f"[PREFIX CACHE] Stored KV cache for {len(all_images)} image(s) ({prompt_tokens_count} prompt tokens)"
                )
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        finish_reason = "length" if generation_tokens >= max_tokens else "stop"

        return MLLMOutput(
            text=output_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
        )

    def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[MLLMOutput]:
        """
        Stream chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            MLLMOutput with incremental text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
        except ImportError:
            # Fallback to non-streaming if stream_generate not available
            output = self.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            yield output
            return

        # Extract text, images, videos, and audio from messages
        chat_messages, all_image_urls, videos, audio_inputs = self._extract_multimodal_messages(messages)
        chat_messages = self._normalize_text_only_messages_for_processor(
            chat_messages,
            has_media=bool(all_image_urls or videos or audio_inputs),
        )

        # Process images
        all_images = []
        all_audio = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))
        if audio_inputs:
            all_audio.extend(self._prepare_audio(audio_inputs))

        # Process videos
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        video_frame_counts: list[int] = []
        for video_path in videos:
            frames = self._prepare_video(
                video_path, fps=video_fps, max_frames=video_max_frames
            )
            all_images.extend(frames)
            video_frame_counts.append(len(frames))

        model_type = (
            str(self.config.get("model_type", "") or "").lower()
            if isinstance(self.config, dict)
            else str(getattr(self.config, "model_type", "") or "").lower()
        )
        if model_type == "gemma4_unified" and video_frame_counts:
            chat_messages = _expand_video_placeholders_to_image_frames(
                chat_messages,
                video_frame_counts,
            )

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        template_tools = kwargs.pop("tools", None)
        enable_thinking = kwargs.pop("enable_thinking", None)
        formatted_prompt = self._apply_chat_template(
            chat_messages,
            enable_thinking,
            tools=template_tools,
        )

        # Post-template image count guard: VLM chat templates may not expand
        # image placeholders for assistant-role messages. Trim all_images to
        # match actual placeholder token count to prevent crashes in generate().
        if all_images and self.config:
            _img_token_id = getattr(self.config, "image_token_index", None)
            if _img_token_id is not None:
                _tok = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                try:
                    _token_ids = _tok.encode(formatted_prompt)
                    _slot_count = _token_ids.count(_img_token_id)
                    if _slot_count != len(all_images):
                        logger.warning(
                            f"Image count mismatch: {len(all_images)} images but "
                            f"{_slot_count} slots in prompt (likely assistant-role "
                            f"images not supported by template). Trimming to {_slot_count}."
                        )
                        all_images = all_images[:_slot_count]
                except Exception as _e:
                    logger.debug(f"Image slot count check failed: {_e}")

        # Check cache for existing KV state (uses images as cache key)
        from mlx_vlm.models import cache as vlm_cache

        prompt_cache = None
        prompt_cache_state = None
        cache_hit = False
        use_cache = kwargs.pop("use_cache", True)
        token_ids: list[int] = []
        media_cache_safe = _supports_simple_media_prompt_cache(self.model)

        if use_cache and media_cache_safe and self._cache_manager is not None and all_images:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = _media_expanded_token_ids_for_cache(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    images=all_images,
                    audio=all_audio,
                    resize_shape=kwargs.get("resize_shape"),
                ) or tokenizer.encode(formatted_prompt)
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                    if prompt_cache_state is None:
                        logger.debug(
                            "Stream chat image prefix hit lacked PromptCacheState; "
                            "falling back to full prompt processing"
                        )
                    cache_hit = True
                    if prefix_match_len > 0:
                        logger.debug(
                            f"Stream chat prefix cache hit: {prefix_match_len} tokens, "
                            f"{len(all_images)} image(s)"
                        )
                    else:
                        logger.debug(f"Stream chat cache hit for {len(all_images)} image(s)")
            except Exception as e:
                logger.debug(f"Stream chat cache fetch failed: {e}")

        # Create new cache if needed
        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Stream generate tokens with cache
        accumulated_text = ""
        token_count = 0
        last_prompt_tokens = 0

        # Ensure processor uses NaiveStreamingDetokenizer for all MLLM models.
        # mlx-vlm's default streaming detokenizer can buffer infinitely for some
        # architectures. NaiveStreamingDetokenizer yields tokens immediately.
        if not hasattr(self.processor, "_patched_detok"):
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

            # Monkey-patch NaiveStreamingDetokenizer to accept kwargs from mlx_vlm
            if not hasattr(NaiveStreamingDetokenizer, "_vml_patched"):
                original_add = NaiveStreamingDetokenizer.add_token
                def _patched_add(self, token, **kwargs):
                    return original_add(self, token)
                NaiveStreamingDetokenizer.add_token = _patched_add
                def _patched_copy(self):
                    return type(self)(self._tokenizer)
                NaiveStreamingDetokenizer.__copy__ = _patched_copy
                NaiveStreamingDetokenizer._vml_patched = True

            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            self.processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
            self.processor._patched_detok = True

        # Extract sampling params not natively supported by mlx_vlm.stream_generate()
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            from ..sampling import make_sampler
            top_p = kwargs.pop("top_p", 1.0)
            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        try:
            with _MaybeVLMStream():
                for chunk in stream_generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    all_images if all_images else None,
                    audio=all_audio if all_audio else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prompt_cache=prompt_cache,
                    **kwargs,
                ):
                    token_count += 1
                    # chunk is a GenerationResult with .text attribute containing the new token
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    accumulated_text += new_text

                    chunk_prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                    if chunk_prompt_tokens:
                        last_prompt_tokens = chunk_prompt_tokens

                    yield MLLMOutput(
                        text=new_text,  # Just the new token for streaming
                        finish_reason=None,
                        prompt_tokens=last_prompt_tokens,
                        completion_tokens=token_count,
                    )
        except Exception as e:
            clear_mlx_memory_cache(log=logger)
            logger.error(f"VLM stream_generate error: {type(e).__name__}: {e}")
            raise

        if (
            use_cache
            and media_cache_safe
            and self._cache_manager is not None
            and all_images
            and not cache_hit
            and prompt_cache
        ):
            try:
                prompt_tokens_count = last_prompt_tokens or len(token_ids)
                cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                    prompt_cache, prompt_tokens_count
                )
                num_img_tokens = 0
                if all_images and self.config:
                    img_token_id = getattr(self.config, "image_token_index", None)
                    if img_token_id is not None and token_ids:
                        num_img_tokens = token_ids.count(img_token_id)
                self._cache_manager.store(
                    images=all_images,
                    prompt=formatted_prompt,
                    vision_embeddings=None,
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=num_img_tokens,
                    model_name=self.model_name,
                )
                logger.info(
                    f"[PREFIX CACHE] Stored streaming KV cache for {len(all_images)} "
                    f"image(s) ({prompt_tokens_count} prompt tokens)"
                )
            except Exception as e:
                logger.warning(f"Failed to cache streaming MLLM prompt: {e}")

        # Final yield with finish_reason
        finish_reason = "length" if token_count >= max_tokens else "stop"
        yield MLLMOutput(
            text="",
            finish_reason=finish_reason,
            prompt_tokens=last_prompt_tokens,
            completion_tokens=token_count,
        )

    def describe_image(
        self,
        image: str,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Convenience method to describe an image.

        Args:
            image: Image path, URL, or base64 string
            prompt: Description prompt
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Image description text
        """
        output = self.generate(
            prompt=prompt,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def answer_about_image(
        self,
        image: str,
        question: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: Image path, URL, or base64 string
            question: Question about the image
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Answer text
        """
        output = self.generate(
            prompt=question,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def describe_video(
        self,
        video: str | dict,
        prompt: str = "Describe what happens in this video.",
        fps: float = 2.0,
        max_frames: int = 32,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Describe a video using frame extraction.

        Args:
            video: Video file path, URL, base64, or OpenAI format dict
            prompt: Description prompt
            fps: Frames per second to extract
            max_frames: Maximum frames to extract
            max_tokens: Maximum tokens to generate

        Returns:
            Video description text

        Example:
            # Local file
            model.describe_video("video.mp4")

            # URL
            model.describe_video("https://example.com/video.mp4")

            # OpenAI format
            model.describe_video({"url": "https://example.com/video.mp4"})
        """
        output = self.generate(
            prompt=prompt,
            videos=[video],
            video_fps=fps,
            video_max_frames=max_frames,
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def get_cache_stats(self) -> dict:
        """
        Get MLLM cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit_rate, tokens_saved, etc.)
        """
        if self._cache_manager is None:
            return {"enabled": False}

        stats = self._cache_manager.get_stats()
        stats["enabled"] = True
        stats["cache_entries"] = len(self._cache_manager)
        stats["max_entries"] = self._cache_manager.max_size
        return stats

    def clear_cache(self) -> None:
        """Clear the MLLM KV cache."""
        if self._cache_manager is not None:
            self._cache_manager.clear()
            logger.info("MLLM cache cleared")

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "type": "multimodal-language-model",
            "supports_video": True,
            "supports_streaming": True,
            "cache_enabled": self.enable_cache,
        }

        if self.config:
            info["model_type"] = getattr(self.config, "model_type", "unknown")

        if self._cache_manager is not None:
            info["cache_stats"] = self._cache_manager.get_stats()

        return info

    @staticmethod
    def list_supported_model_families() -> dict[str, str]:
        """
        List supported model families and their patterns.

        Any model on HuggingFace containing these patterns in the name
        is likely compatible with mlx-vlm.
        """
        return {
            "Qwen-VL": "Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)",
            "LLaVA": "LLaVA vision-language models",
            "Idefics": "Idefics vision-language models",
            "PaliGemma": "PaliGemma multimodal models",
            "Pixtral": "Mistral's Pixtral vision models",
            "Molmo": "Allen AI's Molmo models",
            "Phi-3-Vision": "Microsoft's Phi-3 Vision models",
            "CogVLM": "Tsinghua's CogVLM models",
            "InternVL": "InternVL models",
            "MiniCPM-V": "OpenBMB's MiniCPM-V models",
            "Florence": "Microsoft Florence vision models",
            "DeepSeek-VL": "DeepSeek's vision-language models (DeepSeek-VL, DeepSeek-VL2)",
        }

    @staticmethod
    def is_mllm_model(model_name: str) -> bool:
        """Check if a model name indicates an MLLM model.

        Delegates to the canonical implementation in api.utils which checks
        config.json, model registry, and regex patterns.
        """
        from ..api.utils import is_mllm_model as _canonical
        return _canonical(model_name)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXMultimodalLM model={self.model_name} status={status}>"


# Backwards compatibility aliases
MLXVisionLanguageModel = MLXMultimodalLM
VLMOutput = MLLMOutput
is_vlm_model = MLXMultimodalLM.is_mllm_model

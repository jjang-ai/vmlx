"""MiMo-V2.6 JANGHT text loader (mixed JANGT trellis / JANGH Hadamard routed experts).

Bundle contract (jang_config.json "format": "jangtq2", the published JANGHT runtime keys -- no branded weight_format;
config.json):
  * "jangt" block + per-module "quantization" entries for every routed projection
    (model.layers.L.mlp.switch_mlp.{gate,up,down}_proj -> {"mode": "jangt", "kbits"} | {"mode": "jangtq2", "bits"});
  * every other 2-D text weight (qkv_proj, o_proj, layer-0 dense MLP, embed_tokens, lm_head) affine with the top-level
    group_size / bits; routers, e_score_correction_bias and sinks fp32; media tensors (visual.*, audio_encoder.*,
    speech_embeddings.*) are read by the lazy towers (jang_tools.mimo_v2.v26_omni), never by the text model.
The text math is the stock fresh V2.6 module (jang_tools.mimo_v2.v26_model); only the routed experts change: each MoE's
SwitchGLU becomes a vmlx_engine.jangt JTMixedSwitchGLU and the MoE calls its fused ``routed`` path (fp32 weighted sum,
the reference's accumulation). No process-wide patch: the MoE subclass is assigned per instance.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

MEDIA_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp.")


def is_janght_bundle(contract: dict | None) -> bool:
    """`contract` = read_mimo_v26_contract(bundle), which already checked config.json's routed jangt contract."""
    return isinstance(contract, dict) and contract.get("format") == "jangtq2" and contract.get("weight_format") != "mixed_affine_mxfp4"


def _moe_class():
    from jang_tools.mimo_v2 import v26_model as T

    class JangtMoE(T.MoE):
        """v26 MoE whose routed experts are a JTMixedSwitchGLU (fused gather + weighted combine)."""

        def __call__(self, x):
            idx, w = self.gate(x)
            return self.switch_mlp.routed(x, idx, w).astype(x.dtype)
    return JangtMoE


def load_mimo_v26_janght_text(bundle: str | Path):
    from mlx_lm.utils import load_tokenizer

    bundle = Path(bundle)
    model = build_mimo_v26_janght_model(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    gen = {}
    if (bundle / "generation_config.json").exists():
        gen = json.loads((bundle / "generation_config.json").read_text())
    eos = gen.get("eos_token_id") or cfg.get("eos_token_id")
    tokenizer = load_tokenizer(bundle, eos_token_ids=eos if isinstance(eos, list) else ([eos] if eos is not None else None))
    return model, tokenizer


def build_mimo_v26_janght_model(bundle: str | Path):
    from jang_tools.mimo_v2 import v26_model as T
    from jang_tools.mimo_v2.mlx_register import register
    from vmlx_engine.jangt.switch import install_jangt

    register()
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    q = cfg.get("quantization") or {}
    if not cfg.get("jangt") or not q:
        raise ValueError("MiMo JANGHT bundle without a jangt block / quantization contract")
    args = T.ModelArgs.from_dict(cfg)
    model = T.Model(args)
    n_moe = sum(1 for f in args.moe_layer_freq if f)
    n = install_jangt(model, {"jangt": cfg["jangt"], "quantization": q})
    if n != n_moe:
        raise ValueError(f"MiMo JANGHT: installed {n} routed stacks, the config has {n_moe} MoE layers")
    moe_cls = _moe_class()
    for layer in model.model.layers:
        if isinstance(layer.mlp, T.MoE):
            layer.mlp.__class__ = moe_cls
    index = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted(set(index.values())):
        for k, v in mx.load(str(bundle / shard)).items():
            if not k.startswith(MEDIA_PREFIXES):
                weights[k] = v
    default = {k: q[k] for k in ("group_size", "bits", "mode") if k in q}

    def pred(path, module):
        if ".switch_mlp" in path or not hasattr(module, "to_quantized"):
            return False
        ov = q.get(path)
        if isinstance(ov, dict):
            return ov if ov.get("mode") == "affine" else False
        return default if f"{path}.scales" in weights else False

    nn.quantize(model, class_predicate=pred)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.jangt_modules = n
    return model

"""Load stored affine DFlash2 checkpoints without changing their quantization.

The upstream loader handles floating checkpoints. Packed checkpoints need
quantized modules installed before strict weight loading; floating selector,
normalization and convolution tensors remain exactly as stored.
"""

import json
from dataclasses import fields
from pathlib import Path


def apply_no_rope_layers(draft, cfg) -> list:
    """Honour config ``no_rope_layers``: the upstream drafter rotates every layer, and a NoPE layer trained without rotation must not be."""
    import dflash.model_mlx as runtime

    ids = [int(i) for i in cfg.get("no_rope_layers") or []]

    class _NoRopeAttention(runtime.DFlashAttention):
        def __call__(self, x, x_ctx, rope, cache):
            return super().__call__(x, x_ctx, lambda v, offset=0: v, cache)

    for i in ids:
        draft.layers[i].self_attn.__class__ = _NoRopeAttention
    return ids


# Context window a target family's DFlash2 drafter was trained and is served with, for drafters that have a
# full-attention layer (an all-sliding drafter is bounded by its own sliding window). Ripple-65B: the vendor's server
# runs the draft with --speculative-draft-window-size 16384 and refuses a smaller window (the draft was trained on
# 16K-token rows), so the window may be larger here but never smaller.
DRAFT_CONTEXT_WINDOWS = {"ricursive": 16384}


def draft_context_window(draft, target_type) -> int | None:
    """The draft context window for a drafter with a full-attention layer, or None (unbounded / all-sliding).

    ``VMLX_DFLASH2_DRAFT_WINDOW`` overrides the family value (0 = unbounded, the previous behaviour)."""
    import os

    layer_types = tuple(getattr(draft.config, "layer_types", ()) or ())
    if not layer_types or all(t == "sliding_attention" for t in layer_types):
        return None
    window = DRAFT_CONTEXT_WINDOWS.get(str(target_type or ""))
    raw = os.environ.get("VMLX_DFLASH2_DRAFT_WINDOW", "").strip()
    if raw:
        try:
            window = int(raw)
        except ValueError:
            pass
    if not window or window <= 0:
        return None
    sliding = int(getattr(draft.config, "sliding_window", 0) or 0)
    return max(int(window), sliding)


def apply_draft_context_window(draft, target_type) -> int | None:
    """Bound the drafter's full-attention layers to its context window: their KV becomes a rotating cache of
    ``window - 1`` rows (as the sliding layers keep ``sliding_window - 1``), and the DFlash2 lane feeds at most that
    many prompt hidden rows to a fresh drafter (dflash2_runtime reads ``_vmlx_context_window``). Without it a 128K
    prompt kept every row of five target taps (5 x hidden x 2 bytes per token) until the first draft cycle."""
    window = draft_context_window(draft, target_type)
    if window is None:
        return None
    from mlx_lm.models.cache import RotatingKVCache

    sliding = getattr(draft.config, "sliding_window", None)
    layer_types = tuple(draft.config.layer_types)

    def make_cache():
        return [
            RotatingKVCache(max_size=(sliding if t == "sliding_attention" else window) - 1, keep=0)
            for t in layer_types
        ]

    draft.make_cache = make_cache
    draft._vmlx_context_window = int(window)
    return int(window)


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

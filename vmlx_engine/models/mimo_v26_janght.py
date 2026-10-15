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


# Full-attention prefill (MiMo q/k head_dim 192, v head_dim 128). MLX 0.32's fused attention takes only equal q/k and
# v head dims in {64, 80, 96, 128, 256}, and its full-attention kernel engages only for a causal mask with >= 1024
# query rows; every other call materializes the (heads x rows x keys) fp32 scores. Measured on M5 Max (64 heads,
# 4 kv heads): 2048 rows x 32k keys 478 ms + 17.3 GB transient, x 64k keys 1146 ms + 34.6 GB -- a 128k context
# cannot be prefilled next to the 94 GB model. Zero-padding q/k to 256 and v to 256 (scale stays 192^-0.5; the
# padded value columns are dropped) is the same arithmetic in the fused kernel: 267 ms + 0.27 GB at 32k, 296 ms +
# 0.40 GB at 64k (outputs equal to the unpadded call). A chunk under 1024 rows against a long cache is lifted to
# 1024 rows with leading zero query rows (causal: they only see earlier keys; their outputs are discarded).
# Below the scores budget (decode, DFlash verify rows, short contexts) the native call is as fast or faster.
_FUSED_MIN_ROWS = 1024
_FUSED_HEAD_DIM = 256
_SCORES_BUDGET_BYTES = 1 << 30


# Few query rows (a DFlash verify round, 3-16 rows) against a long cache: MLX's fallback broadcasts the 4 kv heads to
# the 64 query heads and materializes the scores (8 rows x 131k keys: 15.4 ms per full layer vs 1.7 ms for one decode
# row). Folding the 16 query heads that share a kv head into the row axis reads each key once; the softmax with the
# sink logit runs as one compiled fp32 kernel: 8 rows 10.3 ms at 131k, 2.6 ms at 32k (native 4.2); 4 rows 6.9 ms at
# 131k (native 15.6). Two rows stay native (MLX is faster there). Differs from the native call by ~1e-4 (fp32 softmax).
_FEW_ROWS = (3, 16)
_FEW_ROWS_MIN_KEYS = 4096


@mx.compile
def _softmax_with_sinks(s, allowed, sink):
    s = mx.where(allowed, s, -mx.inf)
    m = mx.maximum(mx.max(s, axis=-1, keepdims=True), sink)
    p = mx.exp(s - m)
    return p / (p.sum(axis=-1, keepdims=True) + mx.exp(sink - m))


def _grouped_few_rows(q, k, v, scale, sinks):
    """Causal attention of the last L query rows over T keys, GQA folded into the rows (see _FEW_ROWS)."""
    B, H, L, dq = q.shape
    KV, T, dv = int(k.shape[1]), int(k.shape[2]), int(v.shape[3])
    G = H // KV
    s = ((q * scale).reshape(B, KV, G * L, dq) @ k.transpose(0, 1, 3, 2)).astype(mx.float32).reshape(B, KV, G, L, T)
    allowed = mx.arange(T)[None, :] <= (T - L + mx.arange(L)[:, None])
    sink = (sinks.astype(mx.float32) if sinks is not None else mx.full((H,), -mx.inf)).reshape(1, KV, G, 1, 1)
    p = _softmax_with_sinks(s, allowed, sink)
    return (p.astype(v.dtype).reshape(B, KV, G * L, T) @ v).reshape(B, H, L, dv)


def _full_attention(q, k, v, scale, mask, sinks):
    B, H, L, dq = q.shape
    T, dv = int(k.shape[2]), int(v.shape[3])
    if (_FEW_ROWS[0] <= L <= _FEW_ROWS[1] and T >= _FEW_ROWS_MIN_KEYS and H % int(k.shape[1]) == 0
            and (mask is None or (isinstance(mask, str) and mask == "causal"))):
        return _grouped_few_rows(q, k, v, scale, sinks)
    if (mask is not None and not (isinstance(mask, str) and mask == "causal")) or T < _FUSED_MIN_ROWS \
            or H * L * T * 4 <= _SCORES_BUDGET_BYTES or max(dq, dv) > _FUSED_HEAD_DIM:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)
    lift = max(0, _FUSED_MIN_ROWS - L)
    if lift:
        q = mx.concatenate([mx.zeros((B, H, lift, dq), dtype=q.dtype), q], axis=2)
    pad = lambda a, d: a if d == _FUSED_HEAD_DIM else mx.pad(a, [(0, 0), (0, 0), (0, 0), (0, _FUSED_HEAD_DIM - d)])
    o = mx.fast.scaled_dot_product_attention(pad(q, dq), pad(k, dq), pad(v, dv), scale=scale, mask="causal", sinks=sinks)
    return o[:, :, lift:, :dv]


def _attention_class():
    from jang_tools.mimo_v2 import v26_model as T

    class LongContextAttention(T.Attention):
        """v26 full-attention layer whose attention call stays on MLX's fused kernel at long context (see above)."""

        def __call__(self, x, mask=None, cache=None):
            B, L, _ = x.shape
            qkv = self.qkv_proj(x)
            q, k, v = mx.split(qkv, [self.q_size, self.q_size + self.k_size], axis=-1)
            q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.n_kv, self.v_head_dim).transpose(0, 2, 1, 3)
            if self.v_scale is not None:
                v = v * self.v_scale
            offset = cache.offset if cache is not None else 0
            q = self._rope(q, offset)
            k = self._rope(k, offset)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            sinks = self.attention_sink_bias.astype(q.dtype) if self.has_sink else None
            o = _full_attention(q, k, v, self.scale, mask, sinks)
            return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
    return LongContextAttention


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


def warm_mimo_v26_janght(model) -> float:
    """Build the routed-expert kernels at load instead of on the first request.

    The JANGT trellis / JANGH codebook kernels are Metal source compiled on first use, one variant per format, bit
    width, outlier width and path; measured 2026-10-15 the first request of a fresh process took 8.0-13.5 s to its first
    token on 39-78 token prompts (the same request again: 0.4 s). Three forwards cover the three paths a request takes:
    16 rows (128 expert slots: the sorted prefill tiles), 8 rows (64 slots: the decode kernels at a DFlash verify
    width) and 1 row (decode). A throwaway cache; the allocator cache is released after. VMLX_MIMO_WARMUP=0 skips it.
    Returns the seconds spent (0.0 when skipped)."""
    import logging
    import os
    import time

    if os.environ.get("VMLX_MIMO_WARMUP", "1").strip().lower() in {"0", "false", "off", "no"}:
        return 0.0
    t0 = time.perf_counter()
    cache = model.make_cache()
    for rows in (16, 8, 1):
        mx.eval(model(mx.zeros((1, rows), dtype=mx.int32), cache=cache))
    mx.synchronize()
    del cache
    mx.clear_cache()
    dt = time.perf_counter() - t0
    logging.getLogger(__name__).info("MiMo JANGHT kernels warmed at load (prefill / verify / decode paths): %.1f s", dt)
    return dt


def load_mimo_v26_janght_text(bundle: str | Path):
    from mlx_lm.utils import load_tokenizer

    bundle = Path(bundle)
    model = build_mimo_v26_janght_model(bundle)
    warm_mimo_v26_janght(model)
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
    attn_cls = _attention_class()
    for layer in model.model.layers:
        if isinstance(layer.mlp, T.MoE):
            layer.mlp.__class__ = moe_cls
        if not layer.self_attn.is_swa:
            layer.self_attn.__class__ = attn_cls
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

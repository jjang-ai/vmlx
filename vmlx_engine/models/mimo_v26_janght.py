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
import os
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


_PATHS_SEEN: set = set()


def _note_path(path: str, mask, layer: str = "full") -> None:
    """Log each (attention path, mask kind) once per process: which route a lane's prefill / verify really takes."""
    kind = "none" if mask is None else ("causal" if isinstance(mask, str) else f"array:{mask.dtype}")
    if (layer, path, kind) not in _PATHS_SEEN:
        _PATHS_SEEN.add((layer, path, kind))
        import logging

        logging.getLogger(__name__).info("MiMo %s-attention path: %s (mask %s)", layer, path, kind)


# Native 192/128 NAX attention (metal/nax_attention_bdv.py): MLX's own NAX flash kernel with a separate value head
# dim, no padding, no q/k/v copies, key-range passes at long context; the sliding-window layers get an in-kernel band
# (a tile reads only the ~5 key blocks its rows can see) instead of materialized scores. Declines (returns None) off
# NAX GPUs, for L <= 8, non-bool array masks, and other head dims; the routes below stay as the fallback.
# VMLX_MIMO_NAX_ATTENTION=0 turns it off (A/B only).
_NAX_ATTENTION = os.environ.get("VMLX_MIMO_NAX_ATTENTION", "1").strip().lower() not in {"0", "false", "off", "no"}


# 1-4 query rows over a long cache (decode, DFlash AR steps, verify widths <= 4): split-key "flash decoding" with
# simdgroup matrix products (metal/sdpa_flash_decode.py). MLX has no fused 192/128 vector kernel, so its
# call materializes the scores: one row at 121k keys 1.84 ms per full layer vs 1.12 (x1.64; 9 layers = 6.5 ms per token),
# two rows 3.47 vs 1.54, four rows 23.9 vs 3.11 (synthetic M5 Max, max |err| <= 1.2e-4 vs MLX). Crossovers measured:
# one row from 32k keys, two rows from 4k, three-eight rows from 2k (they also beat _grouped_few_rows: 5-8 rows
# x1.4-4.5 from 8k to 121k keys).
# VMLX_MIMO_FLASH_DECODE=0 turns it off (A/B only).
_FLASH_DECODE = os.environ.get("VMLX_MIMO_FLASH_DECODE", "1").strip().lower() not in {"0", "false", "off", "no"}
_FLASH_MIN_KEYS = {1: 32768, 2: 4096, 3: 2048, 4: 2048, 5: 2048, 6: 2048, 7: 2048, 8: 2048}


def _flash_decode(q, k, v, scale, mask, sinks):
    B, H, L, _ = q.shape
    if not _FLASH_DECODE or int(k.shape[2]) < _FLASH_MIN_KEYS.get(int(L), 1 << 62):
        return None
    from ..metal.sdpa_flash_decode import sdpa_flash

    o = sdpa_flash(q, k, v, scale, mask, sinks)
    if o is None:
        return None
    # (B, L, H * Dv) -> (B, H, L, Dv): the caller's transpose + reshape back to (B, L, H * Dv) is then a view.
    return o.reshape(B, L, H, int(v.shape[3])).transpose(0, 2, 1, 3)


def _nax(q, k, v, scale, mask, sinks, window=0):
    if not _NAX_ATTENTION:
        return None
    from ..metal.nax_attention_bdv import nax_attention_bdv

    return nax_attention_bdv(q, k, v, scale=scale, mask=mask, sinks=sinks, window=window)


def prefill_scores_materialized(model) -> bool:
    """False when every layer's prefill attention (L > 8) runs on the NAX 192/128 kernel: full layers stream keys and
    window layers read their band, so no chunk x context score buffer of prefill-chunk size exists. (A 9-16-row call
    over >= 4096 keys still takes _grouped_few_rows, which materializes H x L x T fp32: ~0.5 GB at 131k keys, ~1/128 of
    the 2048-row buffer the projection sizes.) The batched engine's tight-memory step
    projection sizes chunks by exactly that buffer (prefill_admission.max_prefill_chunk_tokens); for this model it
    capped the plain lane at 1024-row chunks, falling to 519 at 112k context, while the configured step was 2048."""
    try:
        if not _NAX_ATTENTION or not all(isinstance(layer.self_attn, _attention_class.cls)
                                         for layer in model.model.layers):
            return True
        from ..metal import nax_attention_bdv as nab

        return not (nab.available() and nab.self_check_passed())
    except Exception:  # noqa: BLE001 - unknown means "assume materialized" (the existing projection stays)
        return True


def _swa_attention(q, k, v, scale, mask, sinks, window, cache):
    """Sliding-window layer: the model's window mask ("causal" while offset + L <= window, else a bool band over the
    temporally ordered keys that end at the last query) is passed through and the band is applied in-kernel too,
    so the kernel only skips key blocks the mask already excludes."""
    if int(q.shape[2]) > 8 and not hasattr(cache, "bits"):
        o = _nax(q, k, v, scale, mask, sinks, window)
        if o is not None:
            _note_path("nax-bdv-window", mask, layer="sliding-window")
            return o
    from mlx_lm.models.base import scaled_dot_product_attention

    return scaled_dot_product_attention(q, k, v, cache, scale, mask=mask, sinks=sinks)


def _full_attention(q, k, v, scale, mask, sinks):
    B, H, L, dq = q.shape
    T, dv = int(k.shape[2]), int(v.shape[3])
    causal = isinstance(mask, str) and mask == "causal"
    array_mask = isinstance(mask, mx.array) and mask.ndim >= 2 and tuple(mask.shape[-2:]) == (L, T)
    if L <= 8:
        o = _flash_decode(q, k, v, scale, mask, sinks)
        if o is not None:
            _note_path(f"flash-decode-{L}row", mask)
            return o
    if (_FEW_ROWS[0] <= L <= _FEW_ROWS[1] and T >= _FEW_ROWS_MIN_KEYS and H % int(k.shape[1]) == 0
            and (mask is None or causal)):
        _note_path("grouped-few-rows", mask)
        return _grouped_few_rows(q, k, v, scale, sinks)
    if L > 8:
        o = _nax(q, k, v, scale, mask, sinks)
        if o is not None:
            _note_path("nax-bdv", mask)
            return o
    # Array masks (the batched engine's prefill passes explicit masks) are fused as well (measured: 0 GB transient,
    # 2048 x 32k 298-358 ms vs 130 ms for "causal"), so they take the padded path too instead of materializing scores.
    if (mask is not None and not causal and not array_mask) or T < _FUSED_MIN_ROWS \
            or H * L * T * 4 <= _SCORES_BUDGET_BYTES or max(dq, dv) > _FUSED_HEAD_DIM:
        if H * L * T * 4 > _SCORES_BUDGET_BYTES:
            _note_path("native-materialized-scores", mask)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)
    _note_path("padded-fused" + ("-lifted" if L < _FUSED_MIN_ROWS else ""), mask)
    lift = max(0, _FUSED_MIN_ROWS - L)
    if lift:
        q = mx.concatenate([mx.zeros((B, H, lift, dq), dtype=q.dtype), q], axis=2)
        if array_mask:
            # lifted rows see key 0 only (never an all-masked row); their outputs are discarded
            first = mx.arange(T) == 0
            row = first if mask.dtype == mx.bool_ else mx.where(first, 0.0, -mx.inf).astype(mask.dtype)
            pre = mx.broadcast_to(row, (*mask.shape[:-2], lift, T))
            mask = mx.concatenate([pre, mask], axis=-2)
    pad = lambda a, d: a if d == _FUSED_HEAD_DIM else mx.pad(a, [(0, 0), (0, 0), (0, 0), (0, _FUSED_HEAD_DIM - d)])
    o = mx.fast.scaled_dot_product_attention(pad(q, dq), pad(k, dq), pad(v, dv), scale=scale,
                                             mask=mask if array_mask else "causal", sinks=sinks)
    return o[:, :, lift:, :dv]


def _attention_class():
    if _attention_class.cls is not None:
        return _attention_class.cls
    from jang_tools.mimo_v2 import v26_model as T

    class LongContextAttention(T.Attention):
        """v26 attention layer on fused kernels at every context (see above); `_window` > 0 marks a sliding-window
        layer (set at build from config.sliding_window)."""

        _window = 0

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
            if self.is_swa:
                o = _swa_attention(q, k, v, self.scale, mask, sinks, self._window, cache)
            else:
                o = _full_attention(q, k, v, self.scale, mask, sinks)
            return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
    _attention_class.cls = LongContextAttention
    return LongContextAttention


_attention_class.cls = None


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
    t_attn = warm_attention_kernels(model)
    mx.clear_cache()
    dt = time.perf_counter() - t0
    logging.getLogger(__name__).info("MiMo JANGHT kernels warmed at load (prefill / verify / decode paths): %.1f s "
                                     "(long-context attention variants %.1f s)", dt, t_attn)
    return dt


def warm_attention_kernels(model) -> float:
    """Compile the long-context attention kernel variants at load. The three warm-up forwards run on a short cache,
    so the NAX 192/128 variants (aligned / unaligned tiles, key-range passes with the head-dim split, window + bool
    mask + sinks) and the flash decode kernels (1-4 rows) otherwise compile inside the first long request. Suspected
    in the first DFlash chain turn with flash decode on (34.9 tok/s decode vs 53.9 in the flash-off arm, which builds
    none of the flash kernels); not separated from sampling noise on a 19-token answer. Compiling here costs 0.8 s
    at load. Synthetic inputs at the model's head counts; nothing is cached. Returns the seconds spent."""
    import time

    t0 = time.perf_counter()
    try:
        layers = model.model.layers
        full = next((l.self_attn for l in layers if not l.self_attn.is_swa), None)
        swa = next((l.self_attn for l in layers if l.self_attn.is_swa), None)
        dt = mx.bfloat16
        z = lambda *shape: mx.zeros(shape, dtype=dt)
        outs = []
        if full is not None:
            H, KV, dq, dv, sc = full.n_heads, full.n_kv, full.head_dim, full.v_head_dim, full.scale
            for L, T in ((2048, 2048), (100, 3077), (2048, 16389)):  # aligned, unaligned, key-range passes
                outs.append(_full_attention(z(1, H, L, dq), z(1, KV, T, dq), z(1, KV, T, dv), sc, "causal", None))
            causal = (mx.arange(300)[:, None] + 1000) >= mx.arange(1300)[None]  # the batched lane's bool masks
            outs.append(_full_attention(z(1, H, 300, dq), z(1, KV, 1300, dq), z(1, KV, 1300, dv), sc, causal, None))
            for L in range(1, 9):                                       # flash decode rows (DFlash verify widths)
                T = max(_FLASH_MIN_KEYS.get(L, 4096), 4096) + 3
                outs.append(_full_attention(z(1, H, L, dq), z(1, KV, T, dq), z(1, KV, T, dv), sc,
                                            None if L == 1 else "causal", None))
        if swa is not None and getattr(swa, "_window", 0):
            H, KV, dq, dv, sc, W = swa.n_heads, swa.n_kv, swa.head_dim, swa.v_head_dim, swa.scale, swa._window
            sinks = mx.zeros((H,), dtype=dt) if swa.has_sink else None
            for L in (2048, 300):
                T = L + W - 1
                rows = mx.arange(L)[:, None] + (T - L)
                band = (rows >= mx.arange(T)[None]) & (rows - mx.arange(T)[None] < W)
                outs.append(_swa_attention(z(1, H, L, dq), z(1, KV, T, dq), z(1, KV, T, dv), sc, band, sinks, W, None))
            outs.append(_swa_attention(z(1, H, 100, dq), z(1, KV, 100, dq), z(1, KV, 100, dv), sc, "causal", sinks, W,
                                       None))
        mx.eval(outs)
        mx.synchronize()
    except Exception as exc:  # noqa: BLE001 - a warm-up failure only means the first request compiles them
        import logging

        logging.getLogger(__name__).warning("MiMo attention kernel warm-up skipped: %s", exc)
    return time.perf_counter() - t0


def load_mimo_v26_janght_text(bundle: str | Path):
    from mlx_lm.utils import load_tokenizer

    bundle = Path(bundle)
    model = build_mimo_v26_janght_model(bundle)
    # Wire the weights like every other JANG loader (jang_loader._set_wired_limit_for_model). This loader never did, so
    # on SimpleEngine lanes (DFlash on or off; the batched MLLM generator wires during generation) MiMo ran unwired:
    # after >= 1 s idle the next forward paid 0.6-1.0 s (17 rows: 150 -> 750-1185 ms in-process) and the GPU keep-alive
    # could not prevent it; wired, the keep-alive removes it (168-172 ms after 2 / 6 s idle, probe 2026-10-15).
    import os

    if os.environ.get("VMLX_MIMO_WIRE_AT_LOAD", "1").strip().lower() not in {"0", "false", "off", "no"}:
        from ..utils.jang_loader import _set_wired_limit_for_model

        _set_wired_limit_for_model(sorted(bundle.glob("model-*.safetensors")))
    model._vmlx_prefill_scores_materialized = prefill_scores_materialized(model)
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
    # fused routed combine in the sorted prefill (jangt/switch.py, metal/moe_combine.py): MiMo proven, opt-in per model
    from vmlx_engine.jangt.switch import enable_fused_combine

    enable_fused_combine(model)
    moe_cls = _moe_class()
    attn_cls = _attention_class()
    for layer in model.model.layers:
        if isinstance(layer.mlp, T.MoE):
            layer.mlp.__class__ = moe_cls
        layer.self_attn.__class__ = attn_cls
        if layer.self_attn.is_swa:
            layer.self_attn._window = int(args.sliding_window)
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

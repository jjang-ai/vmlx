"""Load-time attention launch fusion for Naive-N0.5-Flash (INTERNAL, vMLX Python).

Decode at T=1 is a dependent chain of small launches (~6.5 µs floor each on M5 Max). Per attention layer this removes:
  * q_proj / k_proj / v_proj -> ONE QuantizedLinear (rows concatenated; all are affine 8-bit g64 in JANGH2/JANGT
    bundles — checked per layer, layers that differ are left alone)
  * `v * attention_value_scale` -> folded into the v rows' affine scales AND biases (w = s*q + b is linear in (s, b))
  * DSA indexer wq / wk / weights_proj (bf16) -> ONE Linear
  * sink bias cast to the activation dtype once, not every call
The original modules are deleted after fusion, so resident memory does not grow. Toggle: N05_FUSED_ATTN=0.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _same_quant(mods) -> bool:
    if not all(isinstance(m, nn.QuantizedLinear) for m in mods):
        return False
    a = mods[0]
    return all((m.bits, m.group_size, getattr(m, "mode", "affine")) == (a.bits, a.group_size, getattr(a, "mode", "affine"))
               and m.weight.shape[1] == a.weight.shape[1] and getattr(m, "biases", None) is not None for m in mods) \
        and getattr(a, "mode", "affine") == "affine" and all("bias" not in m for m in mods)


def fuse_attention(A, value_scale, keep: bool = False) -> bool:
    """Fuse one Attention module in place. Returns True when the q/k/v fusion was applied."""
    if getattr(A, "_fused", False):
        return True
    mods = [A.q_proj, A.k_proj, A.v_proj]
    done = False
    if _same_quant(mods):
        q, k, v = mods
        vs = 1.0 if value_scale is None else float(value_scale)
        W = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
        S = mx.concatenate([q.scales, k.scales, (v.scales.astype(mx.float32) * vs).astype(v.scales.dtype)], axis=0)
        B = mx.concatenate([q.biases, k.biases, (v.biases.astype(mx.float32) * vs).astype(v.biases.dtype)], axis=0)
        f = nn.QuantizedLinear(q.group_size, 4, bias=False, group_size=q.group_size, bits=q.bits)
        f.weight, f.scales, f.biases = W, S, B
        mx.eval(f.weight, f.scales, f.biases)
        A._qkv = f
        A._qkv_split = (q.weight.shape[0], q.weight.shape[0] + k.weight.shape[0])
        A._value_folded = True
        if not keep:
            del A.q_proj, A.k_proj, A.v_proj
        done = True
    ix = getattr(A, "indexer", None)
    if ix is not None and all(type(m) is nn.Linear and "bias" not in m for m in (ix.wq, ix.wk, ix.weights_proj)):
        f = nn.Linear(4, 4, bias=False)
        f.weight = mx.concatenate([ix.wq.weight, ix.wk.weight, ix.weights_proj.weight], axis=0); mx.eval(f.weight)
        ix._fused = f; ix._split = (ix.wq.weight.shape[0], ix.wq.weight.shape[0] + ix.wk.weight.shape[0])
        if not keep:
            del ix.wq, ix.wk, ix.weights_proj
    if A.attention_sink_bias is not None:
        A._sinks_bf16 = A.attention_sink_bias.astype(mx.bfloat16); mx.eval(A._sinks_bf16)
    A._fused = True
    return done


def fuse_model(model, keep: bool = False) -> int:
    """keep=True keeps the original modules (A/B testing only: +attention bytes resident); see set_fused()."""
    vs = model.args.attention_value_scale if hasattr(model, "args") else None
    n = sum(fuse_attention(l.self_attn, vs, keep) for l in model.model.layers)
    mx.clear_cache()
    return n


def set_fused(model, on: bool):
    """A/B toggle for a model fused with keep=True."""
    for l in model.model.layers:
        A = l.self_attn
        if not hasattr(A, "_qkv_store"):
            A._qkv_store = A._qkv
            if A.indexer is not None:
                A.indexer._fused_store = A.indexer._fused
        A._qkv = A._qkv_store if on else None
        A._value_folded = on
        if A.indexer is not None:
            A.indexer._fused = A.indexer._fused_store if on else None

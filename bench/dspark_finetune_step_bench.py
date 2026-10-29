# SPDX-License-Identifier: Apache-2.0
"""Synthetic real-shape timing of one DSpark fine-tune training window (vmlx_engine.dspark.finetune.objective).

MiniMax-M3.1 drafter geometry (hidden 6144, 64 q heads x 128, 4 KV heads, partial RoPE 64 of 128, FFN 12288, Markov
rank 256, sliding window 4096, block 7, 5 taps of 6144), the target's head + embedding at vocab 200,064 quantized
affine 6-bit g64 (as a stand-in for the bundle's), random weights, fp32 masters, bf16 compute, MXFP8 fake
quantization on the decoder linears. A window = M anchors with the full 4096-row context before them. Reports the
median forward + backward seconds of one window over --probes probes after one discarded warm-up, for each --layers
value (1, 2 -> the per-layer slope; the real drafter has 5), plus one AdamW update for the largest layer count.

No real model is loaded. Memory grows ~3.3 GB per layer (fp32 master + gradient) on top of ~5 GB fixed; with
--optimizer another ~2.7 GB per layer. Usage:
  python bench/dspark_finetune_step_bench.py --layers 1,2 --anchors 64 [--probes 5] [--optimizer]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.dspark.finetune import objective as O

HID, HEADS, KV, HD, ROT, FFN, RANK, VOCAB, WINDOW, BLOCK, TAPS = 6144, 64, 4, 128, 64, 12288, 256, 200064, 4096, 7, 5


class _Head:
    def __init__(self, rng):
        mx.random.seed(0)
        self.lm = nn.QuantizedLinear(HID, VOCAB, bias=False, group_size=64, bits=6)
        self.emb = nn.QuantizedEmbedding(VOCAB, HID, group_size=64, bits=6)
        for m in (self.lm, self.emb):                 # random packed codes + small scales: no float copy of the head
            m.weight = mx.random.randint(0, 2 ** 31, (VOCAB, HID * 6 // 32)).astype(mx.uint32)
            m.scales = mx.full((VOCAB, HID // 64), 1e-3, dtype=mx.bfloat16)
            m.biases = mx.full((VOCAB, HID // 64), -0.03, dtype=mx.bfloat16)
        mx.eval(self.lm.parameters(), self.emb.parameters())

    def head(self, h):
        return self.lm(h)

    def embed(self, ids):
        return self.emb(ids)


def _params(layers: int, rng) -> tuple[dict, frozenset]:
    def w(*shape, k=0.02):
        return mx.array(rng.standard_normal(shape, dtype=np.float32) * k)

    p = {"fc.weight": w(HID, TAPS * HID, k=0.005), "hidden_norm.weight": w(HID), "norm.weight": w(HID),
         "markov_head.markov_w1.weight": w(VOCAB, RANK), "markov_head.markov_w2.weight": w(VOCAB, RANK),
         "confidence_head.proj.weight": w(1, HID + RANK), "confidence_head.proj.bias": w(1)}
    fq = set()
    for li in range(layers):
        pre = f"layers.{li}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            p[pre + n + ".weight"] = w(HID)
        for n in ("self_attn.q_norm", "self_attn.k_norm"):
            p[pre + n + ".weight"] = w(HD)
        for n, o, i in (("self_attn.q_proj", HEADS * HD, HID), ("self_attn.k_proj", KV * HD, HID),
                        ("self_attn.v_proj", KV * HD, HID), ("self_attn.o_proj", HID, HEADS * HD),
                        ("mlp.gate_proj", FFN, HID), ("mlp.up_proj", FFN, HID), ("mlp.down_proj", HID, FFN)):
            p[pre + n + ".weight"] = w(o, i)
            fq.add(pre + n)
        mx.eval(p)
    return p, frozenset(fq)


def _spec(layers: int) -> O.Spec:
    return O.Spec(hidden=HID, layers=layers, heads=HEADS, kv_heads=KV, head_dim=HD, rope_dims=ROT, rope_theta=1e7,
                  eps=1e-6, window=WINDOW, block=BLOCK, mask_token_id=VOCAB - 6, taps=tuple(range(TAPS)), vocab=VOCAB,
                  rank=RANK, alpha=1.702, limit=7.0)


def _batch(m: int, rng) -> O.Batch:
    start = 6000
    anchors = start + np.arange(m)
    cs = start - WINDOW + 1
    ce = int(anchors.max())
    feats = mx.array(rng.standard_normal((1, ce - cs, TAPS * HID), dtype=np.float32)).astype(mx.bfloat16)
    toks = rng.integers(0, VOCAB - 10, (m, BLOCK + 1))
    block = np.full((m, BLOCK), VOCAB - 6)
    block[:, 0] = toks[:, 0]
    return O.Batch(feats=feats, ctx_start=cs, anchors=mx.array(anchors.astype(np.int32)),
                   block_tokens=mx.array(block.astype(np.int32)), prev=mx.array(toks[:, :BLOCK].astype(np.int32)),
                   labels=mx.array(toks[:, 1:].astype(np.int32)), weights=mx.ones((m, BLOCK)),
                   valid=mx.ones((m, BLOCK), dtype=mx.bool_), mask=O.attention_mask(anchors, BLOCK, cs, ce - cs, WINDOW),
                   anchors_np=anchors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,2")
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--optimizer", action="store_true")
    ap.add_argument("--no-qat", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    head = _Head(rng)
    b = _batch(a.anchors, rng)
    out = {"anchors": a.anchors, "rows": a.anchors * BLOCK, "ctx_rows": int(b.feats.shape[1]), "qat": not a.no_qat,
           "device": mx.device_info().get("device_name"), "by_layers": {}}
    for layers in [int(x) for x in a.layers.split(",")]:
        p, fq = _params(layers, rng)
        fq = frozenset() if a.no_qat else fq
        spec = _spec(layers)

        def f(tp):
            return O.loss_fn(tp, {}, spec, head, b, fq, candidates=32)

        vg = mx.value_and_grad(f)
        times = []
        for i in range(a.probes + 1):
            t0 = time.perf_counter()
            (loss, _), g = vg(p)
            mx.eval(loss, g)
            if i:
                times.append(time.perf_counter() - t0)
            del g
        row = {"fwd_bwd_s": round(statistics.median(times), 4), "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}
        if a.optimizer:
            import mlx.optimizers as optim

            opt = optim.AdamW(learning_rate=1e-5, weight_decay=0.0)
            (loss, _), g = vg(p)
            mx.eval(g)
            ts = []
            for i in range(3):
                t0 = time.perf_counter()
                p = opt.apply_gradients(g, p)
                mx.eval(p, opt.state)
                ts.append(time.perf_counter() - t0)
            row["adamw_s"] = round(statistics.median(ts), 4)
            row["params_m"] = round(sum(v.size for v in p.values()) / 1e6, 1)
            del opt, g
        out["by_layers"][layers] = row
        print(json.dumps({"layers": layers, **row}), flush=True)
        del p
        mx.clear_cache()
    ls = sorted(out["by_layers"])
    if len(ls) >= 2:
        l0, l1 = ls[0], ls[-1]
        slope = (out["by_layers"][l1]["fwd_bwd_s"] - out["by_layers"][l0]["fwd_bwd_s"]) / (l1 - l0)
        fixed = out["by_layers"][l0]["fwd_bwd_s"] - slope * l0
        out["per_layer_s"], out["fixed_s"] = round(slope, 4), round(fixed, 4)
        out["est_5_layer_window_s"] = round(fixed + 5 * slope, 4)
        out["est_anchors_per_s"] = round(a.anchors / out["est_5_layer_window_s"], 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

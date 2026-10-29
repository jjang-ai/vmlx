"""Synthetic MiniMax-M3 target + MiniMax DSpark drafter for testing the fine-tune pipeline without real weights.

The target is vMLX's own MiniMax-M3 model class at toy size (4 decoder layers: full attention + dense MLP, sparse
attention + dense MLP, two sparse-attention MoE layers with 8 experts top-4 and a shared expert), with random weights
either left in BF16 (the "source") or quantized per module like an M3.1 JANGHT bundle (MXFP8 q / o, affine 8 k / v /
embeddings, affine 6 indexer + head, affine 4 elsewhere, fp32 routers: the "bundle"). The drafter directory is written
in the M3.1 release layout (``language_model.model.dspark.*`` names, MXFP8 e4m3 decoder linears with e8m0
``weight_scale_inv``, BF16 fc / norms / heads) so the release loader, validator and exporter all run on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

D, I, SH, DENSE, E, TOPK, VOCAB = 512, 256, 256, 512, 8, 4, 1024
HEADS, KVH, HD, ROT, LAYERS = 8, 2, 64, 32, 4
MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}
PRE = "language_model.model.dspark."


def target_config(block: int = 32, topk: int = 6) -> dict:
    tc = {"model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": LAYERS, "intermediate_size": I,
          "dense_intermediate_size": DENSE, "shared_intermediate_size": SH, "num_attention_heads": HEADS,
          "num_key_value_heads": KVH, "head_dim": HD, "rotary_dim": ROT, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
          "vocab_size": VOCAB, "num_local_experts": E, "num_experts_per_tok": TOPK, "routed_scaling_factor": 2.0,
          "norm_topk_prob": True, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [0, 0, 1, 1],
          "sparse_attention_config": {"sparse_num_index_heads": KVH, "sparse_index_dim": HD, "sparse_block_size": block,
                                      "sparse_topk_blocks": topk, "sparse_local_block": 1,
                                      "sparse_attention_freq": [0, 1, 1, 1]}}
    return {"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "text_config": tc}


def _bundle_plan(path, module):
    if path.endswith("mlp.gate"):
        return False
    if "switch_mlp" in path:
        return {"group_size": 64, "bits": 4} if hasattr(module, "to_quantized") else False
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    if path.endswith(("q_proj", "o_proj")) and "index" not in path:
        return MX8
    if path.endswith(("k_proj", "v_proj")) and "index" not in path:
        return {"group_size": 64, "bits": 8}
    if "index_" in path or path == "lm_head":
        return {"group_size": 64, "bits": 6}
    if path.endswith("embed_tokens"):
        return {"group_size": 64, "bits": 8}
    return {"group_size": 64, "bits": 4}


def target_model(seed: int = 0, quantize: bool = True, block: int = 32, topk: int = 6, scale: float = 1.5):
    """A tiny MiniMax-M3 (vMLX's model class) with seeded random weights; ``quantize`` = the bundle-like mix."""
    from ...models.minimax_m3 import minimax_m3 as M

    model = M.Model(M.ModelArgs.from_dict(target_config(block, topk)))
    rng = np.random.default_rng(seed)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("norm.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.1, mx.bfloat16
        elif k.endswith("e_score_correction_bias"):
            a, dt = rng.standard_normal(v.shape) * 0.05, mx.float32
        elif k.endswith(".mlp.gate.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.3, mx.float32
        elif "embed_tokens" in k:
            a, dt = rng.standard_normal(v.shape), mx.bfloat16
        else:
            a, dt = rng.standard_normal(v.shape) * scale / np.sqrt(v.shape[-1]), mx.bfloat16
        weights.append((k, mx.array(a.astype(np.float32)).astype(dt)))
    model.load_weights(weights)
    if quantize:
        nn.quantize(model, class_predicate=_bundle_plan)
    mx.eval(model.parameters())
    return model


def drafter_config(*, layers: int = 2, heads: int = 4, kv_heads: int = 2, head_dim: int = 64, ffn: int = 256,
                   rank: int = 16, block: int = 4, window: int = 64, taps=(1, 3)) -> dict:
    text = {"hidden_size": D, "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
            "head_dim": head_dim, "rotary_dim": head_dim // 2, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
            "dspark_ffn_hidden_size": ffn, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "sliding_window": window,
            "dspark_block_size": block, "rope_theta": 10_000.0, "dspark_noise_token_id": VOCAB - 1,
            "dspark_target_layer_ids": list(taps), "dspark_markov_rank": rank,
            "layer_types": ["sliding_attention"] * layers, "dense_intermediate_size": ffn}
    return {"architectures": ["DSparkMiniMaxDraftModel"], "enable_confidence_head": True,
            "confidence_head_with_markov": True, "text_config": text}


def write_release_drafter(path: str | Path, cfg: dict | None = None, seed: int = 0) -> Path:
    """A drafter directory in the release layout with seeded random weights: decoder linears MXFP8 (e4m3 bytes with
    exponent fields 5..8 and e8m0 scales 2^-9..2^-7, i.e. weights of ~1e-3..1e-2), BF16 fc / norms / heads; one shard
    ``model-dspark-00001.safetensors`` + index, as released."""
    from .records import write_safetensors

    path = Path(path)
    cfg = cfg or drafter_config()
    t = cfg["text_config"]
    rng = np.random.default_rng(seed)

    def bf(*shape, k=0.3):
        a = mx.array((rng.standard_normal(shape) * k).astype(np.float32)).astype(mx.bfloat16)
        return ("BF16", np.array(mx.view(a, mx.uint16)))

    def mxfp8(o, i):
        sign = rng.integers(0, 2, (o, i)) << 7
        w = (sign | (rng.integers(5, 9, (o, i)) << 3) | rng.integers(0, 8, (o, i))).astype(np.uint8)
        return ("F8_E4M3", w), ("U8", rng.integers(118, 121, (o, i // 32)).astype(np.uint8))

    hid, rank, ffn = t["hidden_size"], t["dspark_markov_rank"], t["dspark_ffn_hidden_size"]
    heads, kv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    w = {PRE + "fc.weight": bf(hid, len(t["dspark_target_layer_ids"]) * hid, k=0.02),
         PRE + "hidden_norm.weight": bf(hid, k=0.1), PRE + "final_norm.weight": bf(hid, k=0.1),
         PRE + "markov_head.markov_w1.weight": bf(t["vocab_size"], rank, k=0.1),
         PRE + "markov_head.markov_w2.weight": bf(t["vocab_size"], rank, k=0.1),
         PRE + "confidence_head.proj.weight": bf(1, hid + rank, k=0.05), PRE + "confidence_head.proj.bias": bf(1)}
    for L in range(t["num_hidden_layers"]):
        p = f"{PRE}layers.{L}.decoder_layer."
        for n in ("input_layernorm", "post_attention_layernorm"):
            w[p + n + ".weight"] = bf(hid, k=0.1)
        for n in ("self_attn.q_norm", "self_attn.k_norm"):
            w[p + n + ".weight"] = bf(hd, k=0.1)
        for n, o, i in (("self_attn.q_proj", heads * hd, hid), ("self_attn.k_proj", kv * hd, hid),
                        ("self_attn.v_proj", kv * hd, hid), ("self_attn.o_proj", hid, heads * hd),
                        ("mlp.gate_proj", ffn, hid), ("mlp.up_proj", ffn, hid), ("mlp.down_proj", hid, ffn)):
            w[p + n + ".weight"], w[p + n + ".weight_scale_inv"] = mxfp8(o, i)
    path.mkdir(parents=True, exist_ok=True)
    shard = "model-dspark-00001.safetensors"
    size = write_safetensors(path / shard, w, metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": size}, "weight_map": {k: shard for k in sorted(w)}}, indent=2) + "\n")
    (path / "config.json").write_text(json.dumps(cfg, indent=2))
    return path


def bundle_dir(path: str | Path, drafter_src: str | Path) -> Path:
    """A minimal bundle directory for ``load_dspark_runtime``: the target's config.json + ``dspark/`` (a copy)."""
    import shutil

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(target_config()))
    if (root / "dspark").exists():
        shutil.rmtree(root / "dspark")
    shutil.copytree(drafter_src, root / "dspark")
    return root


def random_prompts(n: int, seed: int, lo: int = 120, hi: int = 300, vocab: int = VOCAB) -> list:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab - 1, int(rng.integers(lo, hi))).tolist() for _ in range(n)]

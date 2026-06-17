# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 draft head for MiniMax-M3 speculative decoding (drop-in for vMLX).

Loads the bundled sidecar (`eagle3_config.json` + `eagle3_runtime.safetensors`) and
exposes a KV-cached draft that proposes a token chain from the target's hidden states.
Proven recurrence (== vLLM llama_eagle3.py); accept ~1.76x on REAP22+JANG_2L (greedy).

Integration (see M3-EAGLE3-IMPLEMENTATION-SPEC.md):
  1. target forward with return_aux=True -> capture hidden @ aux_layers (output residual).
  2. draft.propose(target_aux_at_p, last_token, embed_fn, K, sample_fn) -> [d0..dK-1].
  3. verify the chain in one target forward; accept longest matching prefix; emit +1 bonus.
  4. roll back the MSA dual cache (keys,values,idx_keys,offset) to (p+accept+1) — positional, exact.

Created by Jinho Jang (eric@jangq.ai). Head trained by Inferact (TorchSpec), MiniMax-M3 license.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


def _rmsnorm(x, w, eps=1e-6):
    x = x.astype(mx.float32)
    n = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return (w * n).astype(w.dtype)


class Eagle3Draft:
    """KV-cached single-layer EAGLE3 draft. One new token per step()."""

    def __init__(self, weights: dict, cfg: dict):
        self.w = weights
        self.H = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.hd = cfg["head_dim"]
        self.theta = float(cfg["rope_theta"])
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.naux = len(cfg["aux_hidden_state_layers"])
        self.k = self.v = None
        self.n = 0

    # ---- draft KV management ----
    def reset(self):
        self.k = self.v = None; self.n = 0

    def truncate(self, length: int):
        self.n = length
        if length == 0:
            self.k = self.v = None
        else:
            self.k = self.k[:, :, :length]; self.v = self.v[:, :, :length]

    # ---- fuse the 3 target aux hidden states ----
    def combine(self, aux):                       # aux: [...,naux*H]
        w = self.w
        chunks = mx.split(aux, self.naux, axis=-1)
        fused = mx.concatenate(
            [_rmsnorm(chunks[i], w[f"fc_norm.{i}.weight"], self.eps) for i in range(self.naux)],
            axis=-1)
        return fused @ w["fc.weight"].T

    # ---- one draft step at the current cached position ----
    def step(self, embed_vec, fused_vec):         # [1,1,H],[1,1,H] -> (feat,logits)
        w = self.w; pos = self.n
        e = _rmsnorm(embed_vec, w["layers.0.input_layernorm.weight"], self.eps)
        hs = _rmsnorm(fused_vec, w["layers.0.hidden_norm.weight"], self.eps)
        x = mx.concatenate([e, hs], axis=-1)
        q = (x @ w["layers.0.self_attn.q_proj.weight"].T).reshape(1, 1, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = (x @ w["layers.0.self_attn.k_proj.weight"].T).reshape(1, 1, self.nh, self.hd).transpose(0, 2, 1, 3)
        v = (x @ w["layers.0.self_attn.v_proj.weight"].T).reshape(1, 1, self.nh, self.hd).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos)
        k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos)
        self.k = k if self.k is None else mx.concatenate([self.k, k], axis=2)
        self.v = v if self.v is None else mx.concatenate([self.v, v], axis=2)
        self.n += 1
        o = mx.fast.scaled_dot_product_attention(q, self.k, self.v, scale=1.0 / math.sqrt(self.hd))
        o = o.transpose(0, 2, 1, 3).reshape(1, 1, self.nh * self.hd)
        h = fused_vec + o @ w["layers.0.self_attn.o_proj.weight"].T
        hn = _rmsnorm(h, w["layers.0.post_attention_layernorm.weight"], self.eps)
        gu = (nn.silu(hn @ w["layers.0.mlp.gate_proj.weight"].T) * (hn @ w["layers.0.mlp.up_proj.weight"].T))
        h = h + gu @ w["layers.0.mlp.down_proj.weight"].T
        logits = _rmsnorm(h, w["norm.weight"], self.eps) @ w["lm_head.weight"].T
        return h, logits

    # ---- propose a chain of K tokens from verified position p ----
    def propose(self, target_aux_p, last_token, embed_fn, K, sample_fn=None):
        """target_aux_p: [1,1,naux*H]; last_token: int; embed_fn(id)->[1,1,H].
        Returns list of K proposed token ids. sample_fn(logits)->id (default argmax)."""
        if sample_fn is None:
            sample_fn = lambda lg: int(mx.argmax(lg[0, -1]))
        self.reset()
        feat, logits = self.step(embed_fn(last_token), self.combine(target_aux_p))
        out = [sample_fn(logits)]
        for _ in range(K - 1):
            feat, logits = self.step(embed_fn(out[-1]), feat)
            out.append(sample_fn(logits))
        return out


def load_eagle3_draft(bundle_dir, embed_quantize: bool = False):
    """Load the bundled EAGLE3 sidecar. Returns (Eagle3Draft, cfg).

    embed_quantize: if your engine quantizes the draft on load, do it on `weights`
    before constructing (keep lm_head/fc/layer >=6-bit; 2-bit tanks accept)."""
    bundle_dir = Path(bundle_dir)
    cfg = json.loads((bundle_dir / "eagle3_config.json").read_text())
    weights = mx.load(str(bundle_dir / cfg["weights_file"]))
    return Eagle3Draft(weights, cfg), cfg

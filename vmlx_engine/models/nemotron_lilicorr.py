# SPDX-License-Identifier: Apache-2.0
"""NVIDIA LiLiCorr drafter for Nemotron-3.5-Bolt (``lilicorr/`` in the bundle), MLX port.

Semantics follow SGLang's merged LiLiCorr (sgl-project/sglang#37462: srt/models/lilicorr.py, srt/models/dflash.py,
srt/speculative/lilicorr_utils.py, kernels/ops/speculative/lilicorr.py), the serving code the checkpoint ships for:

Drafter body (DFlashDraftModel):
  * context feature = hidden_norm(fc(concat(target taps))), taps = OUTPUTS of target blocks dflash_config.target_layer_ids
    (absolute indices of the 88-block stack, before the final norm); context K/V per draft layer = rope(k_norm(k_proj(ctx)))
    and v_proj(ctx), no conv, no layer norm (the projected feature feeds every layer);
  * block = [anchor token, mask x (block_size - 1)] embedded with the TARGET embedding table (mask_token_id 990);
  * five pre-norm layers: input_layernorm -> grouped dynamic conv (prepare) -> attention (per-head q/k RMSNorm, neox
    RoPE theta 1e4, per-head attention_sink_bias, NON-causal inside a left window of sliding_window) -> conv (finish) ->
    residual; post_attention_layernorm -> conv prepare -> SwiGLU MLP -> conv finish -> residual; final norm;
  * grouped dynamic conv: K taps over the block's own positions, coefficients = base_kernel[side] + a per-token delta
    from kernel_projection(input) (groups of conv_group_size channels), tap t reads position p - t only when p mod
    block_size >= t.
LiLiCorr head (LiLiCorrHead.score + lilicorr_sample_path):
  * candidates per draft slot = top-k of log_softmax(lm_head(draft hidden)) over the FULL vocabulary;
  * candidate state = token_proj(target_embed(candidate)) + pass_hidden_proj(draft hidden) + feature_mlp(5 features:
    log p, p, log p - max, rank fraction, is-top1) + slot / rank embeddings; two lattice layers (relative-slot +
    same-slot attention bias); anchor = context_proj(context feature at the last committed position);
  * factor edges: start = <anchor_out, in_vec(slot 0)>, pair = <out_vec(slot s), in_vec(slot s + 1)>, x logit_scale;
    the path is walked left to right (greedy: argmax; sampled: softmax at the request temperature).
NVFP4 modules: y = (W_nvfp4 x) * weight_scale_2 (modelopt's per-tensor global scale) [+ bias].
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class NVFP4Linear(nn.Module):
    def __init__(self, weight, scales, scale_2, bias=None, group_size=16):
        super().__init__()
        self.weight, self.scales, self.weight_scale_2 = weight, scales, scale_2.astype(mx.float32)
        if bias is not None:
            self.bias = bias
        self.group_size = group_size

    def __call__(self, x):
        y = mx.quantized_matmul(x, self.weight, self.scales, None, transpose=True, group_size=self.group_size,
                                bits=4, mode="nvfp4")
        y = y * self.weight_scale_2.astype(y.dtype)
        if "bias" in self:
            y = y + self.bias
        return y


def _linear(w: dict, name: str, group_size: int = 16):
    if f"{name}.weight_scale_2" in w:
        return NVFP4Linear(w[f"{name}.weight"], w[f"{name}.scales"], w[f"{name}.weight_scale_2"], w.get(f"{name}.bias"),
                           group_size)
    weight = w[f"{name}.weight"]
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=f"{name}.bias" in w)
    lin.weight = weight
    if f"{name}.bias" in w:
        lin.bias = w[f"{name}.bias"]
    return lin


def _rms(w, name, eps):
    n = nn.RMSNorm(w[f"{name}.weight"].shape[0], eps=eps)
    n.weight = w[f"{name}.weight"]
    return n


class GroupedConv(nn.Module):
    def __init__(self, w, name, block_size, taps, group_size):
        super().__init__()
        self.base_kernel = w[f"{name}.base_kernel"]                     # (2, taps, D)
        self.kernel_projection = _linear(w, f"{name}.kernel_projection")
        self.block, self.taps, self.gs = int(block_size), int(taps), int(group_size)

    def _convolve(self, x, delta, side):
        # x (N, D) flattened block rows; delta (N, taps, G)
        N, D = x.shape
        G = D // self.gs
        blocks = x.reshape(N, G, self.gs)
        coeff = self.base_kernel[side].reshape(1, self.taps, G, self.gs) + delta[..., None]
        out = coeff[:, 0] * blocks
        pos = mx.arange(N) % self.block
        for t in range(1, self.taps):
            shifted = mx.concatenate([mx.zeros((t, G, self.gs), dtype=blocks.dtype), blocks[:-t]], axis=0)
            out = out + coeff[:, t] * shifted * (pos >= t).astype(blocks.dtype).reshape(-1, 1, 1)
        return out.reshape(N, D)

    def prepare(self, x):
        N, D = x.shape
        c = self.kernel_projection(x).reshape(N, 2, self.taps, D // self.gs)
        return self._convolve(x, c[:, 0], 0), c[:, 1]

    def finish(self, x, coeff):
        return self._convolve(x, coeff, 1)


class DraftAttention(nn.Module):
    def __init__(self, w, name, c):
        super().__init__()
        self.H, self.KV, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        eps = c["rms_norm_eps"]
        self.q_proj, self.k_proj = _linear(w, f"{name}.q_proj"), _linear(w, f"{name}.k_proj")
        self.v_proj, self.o_proj = _linear(w, f"{name}.v_proj"), _linear(w, f"{name}.o_proj")
        self.q_norm, self.k_norm = _rms(w, f"{name}.q_norm", eps), _rms(w, f"{name}.k_norm", eps)
        self.sinks = w[f"{name}.attention_sink_bias"]
        self.theta = float(c.get("rope_theta", 10000.0))
        self.scale = self.hd ** -0.5

    def rope(self, x, pos0):
        return mx.fast.rope(x, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos0)

    def ctx_kv(self, ctx, pos0):
        n = ctx.shape[1]
        k = self.k_norm(self.k_proj(ctx).reshape(1, n, self.KV, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(ctx).reshape(1, n, self.KV, self.hd).transpose(0, 2, 1, 3)
        return self.rope(k, pos0), v

    def __call__(self, h, kc, vc, pos0, mask):
        B, S, _ = h.shape
        q = self.q_norm(self.q_proj(h).reshape(B, S, self.H, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(h).reshape(B, S, self.KV, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, S, self.KV, self.hd).transpose(0, 2, 1, 3)
        q, k = self.rope(q, pos0), self.rope(k, pos0)
        if kc is not None:
            k, v = mx.concatenate([kc, k], axis=2), mx.concatenate([vc, v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask, sinks=self.sinks.astype(q.dtype))
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, S, -1))


class DraftLayer(nn.Module):
    def __init__(self, w, i, c, block):
        super().__init__()
        p, dc, eps = f"layers.{i}", c["dflash_config"], c["rms_norm_eps"]
        self.input_layernorm = _rms(w, f"{p}.input_layernorm", eps)
        self.post_attention_layernorm = _rms(w, f"{p}.post_attention_layernorm", eps)
        self.self_attn = DraftAttention(w, f"{p}.self_attn", c)
        self.gate_proj, self.up_proj = _linear(w, f"{p}.mlp.gate_proj"), _linear(w, f"{p}.mlp.up_proj")
        self.down_proj = _linear(w, f"{p}.mlp.down_proj")
        taps, gs = int(dc.get("conv_kernel_size") or 0), int(dc.get("conv_group_size") or 0)
        self.attention_conv = GroupedConv(w, f"{p}.attention_conv", block, taps, gs) if taps else None
        self.mlp_conv = GroupedConv(w, f"{p}.mlp_conv", block, taps, gs) if taps else None

    def __call__(self, h, kc, vc, pos0, mask):
        B, S, D = h.shape
        x = self.input_layernorm(h).reshape(B * S, D)
        kern = None
        if self.attention_conv is not None:
            x, kern = self.attention_conv.prepare(x)
        a = self.self_attn(x.reshape(B, S, D), kc, vc, pos0, mask).reshape(B * S, D)
        if kern is not None:
            a = self.attention_conv.finish(a, kern)
        h = h + a.reshape(B, S, D)
        x = self.post_attention_layernorm(h).reshape(B * S, D)
        kern = None
        if self.mlp_conv is not None:
            x, kern = self.mlp_conv.prepare(x)
        m = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        if kern is not None:
            m = self.mlp_conv.finish(m, kern)
        return h + m.reshape(B, S, D)


class LatticeLayer(nn.Module):
    def __init__(self, w, p, heads, eps):
        super().__init__()
        self.attn_norm, self.mlp_norm = _rms(w, f"{p}.attn_norm", eps), _rms(w, f"{p}.mlp_norm", eps)
        self.in_w, self.in_b = w[f"{p}.attn.in_proj_weight"], w[f"{p}.attn.in_proj_bias"]
        self.out_proj = _linear(w, f"{p}.attn.out_proj")
        self.up, self.down = _linear(w, f"{p}.mlp.up_proj"), _linear(w, f"{p}.mlp.down_proj")
        self.heads = heads

    def __call__(self, h, bias):
        B, L, D = h.shape
        x = self.attn_norm(h)
        qkv = x @ self.in_w.T + self.in_b
        q, k, v = mx.split(qkv, 3, axis=-1)
        hd = D // self.heads
        q, k, v = (t.reshape(B, L, self.heads, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=bias)
        h = h + self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))
        return h + self.down(nn.silu(self.up(self.mlp_norm(h))))


def _l2n(x, eps):
    n = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    return (x.astype(mx.float32) / mx.maximum(n, eps))


class LiLiCorrHead(nn.Module):
    def __init__(self, w, c, block):
        super().__init__()
        dc, eps = c["dflash_config"], c["rms_norm_eps"]
        p = "lilicorr"
        self.topk, self.slots = int(dc["lilicorr_candidate_topk"]), int(block) - 1
        self.H, self.hidden = int(dc["lilicorr_num_heads"]), int(dc["lilicorr_hidden_size"])
        self.eps, self.logit_scale = float(dc["lilicorr_vector_eps"]), float(dc["lilicorr_logit_scale"])
        self.token_proj, self.pass_hidden_proj = _linear(w, f"{p}.token_proj"), _linear(w, f"{p}.pass_hidden_proj")
        self.feature_norm_w, self.feature_norm_b = w[f"{p}.feature_norm.weight"], w[f"{p}.feature_norm.bias"]
        self.feature_up, self.feature_down = _linear(w, f"{p}.feature_mlp.up_proj"), _linear(w, f"{p}.feature_mlp.down_proj")
        self.slot_embedding, self.rank_embedding = w[f"{p}.slot_embedding"], w[f"{p}.rank_embedding"]
        self.context_proj = _linear(w, f"{p}.context_proj")
        n_layers = int(dc["lilicorr_num_layers"])
        self.layers = [LatticeLayer(w, f"{p}.layers.{i}", self.H, eps) for i in range(n_layers)]
        self.output_norm, self.anchor_norm = _rms(w, f"{p}.output_norm", eps), _rms(w, f"{p}.anchor_norm", eps)
        fw = w[f"{p}.factor_input_proj.weight"]
        hd = self.hidden
        self.f_self, self.f_anchor, self.f_cross = fw[:, :hd], fw[:, hd:2 * hd], fw[:, 2 * hd:]
        self.f_bias = w[f"{p}.factor_input_proj.bias"]
        self.out_head, self.in_head = _linear(w, f"{p}.out_head"), _linear(w, f"{p}.in_head")
        self.anchor_out_head = _linear(w, f"{p}.anchor_out_head")
        # attention bias [heads, L, L] over (slot, rank) rows: relative-slot + same-slot
        rel_bias, same = np.array(w[f"{p}.relative_slot_bias"].astype(mx.float32)), np.array(
            w[f"{p}.same_slot_bias"].astype(mx.float32))
        slot_ids = np.repeat(np.arange(self.slots), self.topk)
        rel = np.clip(slot_ids[:, None] - slot_ids[None, :], -(block - 1), block - 1)
        bias = rel_bias[:, rel + block - 1] + (slot_ids[:, None] == slot_ids[None, :])[None] * same[:, None, None]
        self.attn_bias = mx.array(bias[None].astype(np.float32))
        rank_frac = np.arange(self.topk, dtype=np.float32) / max(1, self.topk - 1)
        self.rank_frac = mx.array(rank_frac)
        self.is_top1 = mx.array(np.eye(1, self.topk, dtype=np.float32)[0])

    def score(self, token_emb, cand_logp, pass_hidden, anchor_ctx):
        """token_emb (S, K, Dm), cand_logp (S, K) fp32, pass_hidden (S, Dm), anchor_ctx (Dm,) -> start (K,), pair (S-1, K, K)."""
        S, K = cand_logp.shape
        dtype = self.attn_bias.dtype
        tok = self.token_proj(token_emb.astype(mx.bfloat16))
        pas = self.pass_hidden_proj(pass_hidden.astype(mx.bfloat16))[:, None, :]
        lp = cand_logp.astype(mx.float32)
        feats = mx.stack([lp, mx.exp(lp), lp - mx.max(lp, axis=-1, keepdims=True),
                          mx.broadcast_to(self.rank_frac, lp.shape), mx.broadcast_to(self.is_top1, lp.shape)], axis=-1)
        feats = mx.fast.layer_norm(feats.astype(mx.bfloat16), self.feature_norm_w, self.feature_norm_b, 1e-5)
        fm = self.feature_down(nn.silu(self.feature_up(feats)))
        h = tok + pas + fm + self.slot_embedding[0, 0, :S] + self.rank_embedding[0, 0]
        h = h.reshape(1, S * K, self.hidden)
        bias = self.attn_bias[:, :, : S * K, : S * K].astype(h.dtype)
        for layer in self.layers:
            h = layer(h, bias)
        h = self.output_norm(h).reshape(S, K, self.hidden)
        anchor = self.anchor_norm(self.context_proj(anchor_ctx.astype(mx.bfloat16)[None]))[0]
        pre = h @ self.f_self.T + self.f_bias + anchor @ self.f_anchor.T + (h * anchor) @ self.f_cross.T
        fh = nn.silu(pre)
        out_vec, in_vec = _l2n(self.out_head(fh), self.eps), _l2n(self.in_head(fh), self.eps)
        anchor_out = _l2n(self.anchor_out_head(anchor[None]), self.eps)[0]
        start = (in_vec[0] * anchor_out).sum(-1) * self.logit_scale                  # (K,)
        pair = (out_vec[:-1] @ in_vec[1:].transpose(0, 2, 1)) * self.logit_scale     # (S-1, K, K)
        return start, pair


def greedy_walk(start, pair, cand_ids):
    """Left-to-right lattice walk (greedy): slot 0 by start score, slot s by pair[s-1][previous]."""
    start, pair, cand = np.array(start), np.array(pair), np.array(cand_ids)
    prev = int(np.argmax(start))
    out = [int(cand[0, prev])]
    for s in range(pair.shape[0]):
        prev = int(np.argmax(pair[s, prev]))
        out.append(int(cand[s + 1, prev]))
    return out


class LiLiCorrDrafter(nn.Module):
    def __init__(self, path):
        super().__init__()
        path = Path(path)
        c = json.loads((path / "config.json").read_text())
        dc = c["dflash_config"]
        w = {}
        for f in sorted(path.glob("*.safetensors")):
            w.update(mx.load(str(f)))
        self.cfg = c
        self.block = int(c.get("block_size", 8))
        self.layer_ids = list(dc["target_layer_ids"])
        self.mask_id = int(dc["mask_token_id"])
        self.window = int(dc.get("swa_window_size") or c.get("sliding_window") or 1 << 30)
        self.fc = _linear(w, "fc")
        self.hidden_norm = _rms(w, "hidden_norm", c["rms_norm_eps"])
        self.layers = [DraftLayer(w, i, c, self.block) for i in range(c["num_hidden_layers"])]
        self.norm = _rms(w, "norm", c["rms_norm_eps"])
        self.lilicorr = LiLiCorrHead(w, c, self.block) if dc.get("lilicorr_enabled") else None
        mx.eval(self.parameters())
        self.reset()

    def reset(self):
        self.kv = [(None, None) for _ in self.layers]
        self.ctx_end = 0
        self.last_ctx = None

    def add_context(self, taps, pos0):
        """taps (n, n_layers*D) for target positions pos0.. (appended after the current context)."""
        assert pos0 == self.ctx_end, (pos0, self.ctx_end)
        ctx = self.hidden_norm(self.fc(taps[None].astype(mx.bfloat16)))
        self.last_ctx = ctx[0, -1]
        new = []
        for L, (kc, vc) in zip(self.layers, self.kv):
            k, v = L.self_attn.ctx_kv(ctx, pos0)
            if kc is not None:
                k, v = mx.concatenate([kc, k], axis=2), mx.concatenate([vc, v], axis=2)
            if k.shape[2] > self.window:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            new.append((k, v))
        self.kv = new
        self.ctx_end = pos0 + taps.shape[0]
        # bounded graph without a host wait: the next draft_hidden reads these on the same stream
        mx.async_eval([x for kv in self.kv for x in kv], self.last_ctx)

    def draft_hidden(self, block_emb, pos0):
        """block_emb (1, B, D) at positions pos0.. (pos0 == ctx_end) -> final hidden (1, B, D)."""
        assert pos0 == self.ctx_end, (pos0, self.ctx_end)
        B = block_emb.shape[1]
        n = self.kv[0][0].shape[2] if self.kv[0][0] is not None else 0
        kpos = np.concatenate([np.arange(pos0 - n, pos0), np.arange(pos0, pos0 + B)])
        qpos = np.arange(pos0, pos0 + B)
        mask = mx.array(kpos[None, :] > qpos[:, None] - self.window)[None, None]
        h = block_emb
        for L, (kc, vc) in zip(self.layers, self.kv):
            h = L(h, kc, vc, pos0, mask)
        return self.norm(h)

    def propose(self, hidden, lm_head, embed, use_lilicorr=True):
        """hidden (1, B, D) -> (lilicorr path tokens, plain top-1 tokens) for slots 1..B-1."""
        pass_h = hidden[0, 1:]
        logits = lm_head(pass_h).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top1 = mx.argmax(logp, axis=-1)
        if self.lilicorr is None or not use_lilicorr:
            mx.eval(top1)
            return None, top1.tolist()
        K = self.lilicorr.topk
        # top-K in descending order (the head's rank features need rank order): partial selection, then sort the K,
        # instead of a full sort of the 131k vocabulary per slot
        part = mx.argpartition(-logp, kth=K - 1, axis=-1)[:, :K]
        part_lp = mx.take_along_axis(logp, part, axis=-1)
        order = mx.argsort(-part_lp, axis=-1)
        ids = mx.take_along_axis(part, order, axis=-1)
        cand_lp = mx.take_along_axis(part_lp, order, axis=-1)
        start, pair = self.lilicorr.score(embed(ids), cand_lp, pass_h, self.last_ctx)
        # one device sync for everything the host walk reads (np.array on each of them was 3-4 separate waits)
        mx.eval(start, pair, ids, top1)
        return greedy_walk(start, pair, ids), top1.tolist()

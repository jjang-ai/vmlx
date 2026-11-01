# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter, MiniMax-M3.1 flavor: block forward vs a NumPy reference.

The reference is written from the draft's definition: context = fc(concat of
target taps) -> Gemma (1 + w) norm, projected per layer by k_proj / v_proj with a
per-head Gemma k_norm and partial RoPE at the context positions; the block
[anchor, noise...] attends non-causally to itself and to context keys inside the
sliding window (key position > query position - window); MiniMax layers (Gemma
norms, per-head q/k norms, SwiGLU-OAI dense FFN); final Gemma norm. The context
is longer than the window, so some keys are excluded for some rows and not
others, which the test asserts before comparing.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

HID, HEADS, KV, HD, ROT, FFN, TAPS = 32, 4, 2, 8, 4, 48, 2
WINDOW, BLOCK, CTX = 6, 4, 9
ALPHA, LIMIT, EPS, THETA = 1.702, 7.0, 1e-6, 10_000.0
REL_TOL = 1e-2  # TF32 float32 matmuls on M5 (see test_minimax_m3_msa_per_group.py)


def _raw_config():
    return {"architectures": ["DSparkMiniMaxDraftModel"], "enable_confidence_head": True,
            "confidence_head_with_markov": True, "text_config": {
                "hidden_size": HID, "num_hidden_layers": 2, "num_attention_heads": HEADS,
                "num_key_value_heads": KV, "head_dim": HD, "rotary_dim": ROT, "rms_norm_eps": EPS,
                "vocab_size": 64, "dspark_ffn_hidden_size": FFN, "swiglu_alpha": ALPHA, "swiglu_limit": LIMIT,
                "sliding_window": WINDOW, "dspark_block_size": BLOCK, "rope_theta": THETA,
                "dspark_noise_token_id": 63, "dspark_target_layer_ids": [1, 3], "dspark_markov_rank": 8,
                "layer_types": ["sliding_attention"] * 2, "dense_intermediate_size": FFN}}


def _weights(rng):
    w = lambda *s: (rng.standard_normal(s) * 0.3).astype(np.float32)  # noqa: E731
    out = {"fc.weight": w(HID, TAPS * HID), "hidden_norm.weight": w(HID), "norm.weight": w(HID),
           "markov_head.markov_w1.weight": w(64, 8), "markov_head.markov_w2.weight": w(64, 8),
           "confidence_head.proj.weight": w(1, HID + 8), "confidence_head.proj.bias": w(1)}
    for L in range(2):
        p = f"layers.{L}."
        out.update({p + "input_layernorm.weight": w(HID), p + "post_attention_layernorm.weight": w(HID),
                    p + "self_attn.q_proj.weight": w(HEADS * HD, HID), p + "self_attn.k_proj.weight": w(KV * HD, HID),
                    p + "self_attn.v_proj.weight": w(KV * HD, HID), p + "self_attn.o_proj.weight": w(HID, HEADS * HD),
                    p + "self_attn.q_norm.weight": w(HD), p + "self_attn.k_norm.weight": w(HD),
                    p + "mlp.gate_proj.weight": w(FFN, HID), p + "mlp.up_proj.weight": w(FFN, HID),
                    p + "mlp.down_proj.weight": w(HID, FFN)})
    return out


def _gnorm(x, w):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS) * (1.0 + w)


def _rope(x, pos):  # x [S, heads, HD]
    half, inv = ROT // 2, THETA ** (-np.arange(0, ROT, 2) / ROT)
    ang = pos[:, None, None] * inv[None, None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:ROT]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, x[..., ROT:]], axis=-1)


def _reference(W, feats, block, window=WINDOW):
    ctx = _gnorm(feats @ W["fc.weight"].T, W["hidden_norm.weight"])
    cpos, bpos = np.arange(CTX, dtype=np.float64), np.arange(CTX, CTX + BLOCK, dtype=np.float64)
    h = block.astype(np.float64)
    for L in range(2):
        p = f"layers.{L}.self_attn."
        x = _gnorm(h, W[f"layers.{L}.input_layernorm.weight"])
        ck = _rope(_gnorm((ctx @ W[p + "k_proj.weight"].T).reshape(CTX, KV, HD), W[p + "k_norm.weight"]), cpos)
        cv = (ctx @ W[p + "v_proj.weight"].T).reshape(CTX, KV, HD)
        q = _rope(_gnorm((x @ W[p + "q_proj.weight"].T).reshape(BLOCK, HEADS, HD), W[p + "q_norm.weight"]), bpos)
        bk = _rope(_gnorm((x @ W[p + "k_proj.weight"].T).reshape(BLOCK, KV, HD), W[p + "k_norm.weight"]), bpos)
        bv = (x @ W[p + "v_proj.weight"].T).reshape(BLOCK, KV, HD)
        k, v = np.concatenate([ck, bk]), np.concatenate([cv, bv])
        kpos = np.concatenate([cpos, np.full(BLOCK, np.inf)])  # block keys: always visible (non-causal)
        keep = (kpos[None, :] > bpos[:, None] - window) if window else np.ones((BLOCK, CTX + BLOCK), bool)
        out = np.zeros((BLOCK, HEADS, HD))
        for hh in range(HEADS):
            kv = hh // (HEADS // KV)
            lg = np.where(keep, (q[:, hh] @ k[:, kv].T) * HD ** -0.5, -np.inf)
            pr = np.exp(lg - lg.max(-1, keepdims=True))
            out[:, hh] = (pr / pr.sum(-1, keepdims=True)) @ v[:, kv]
        h = h + out.reshape(BLOCK, -1) @ W[p + "o_proj.weight"].T
        y = _gnorm(h, W[f"layers.{L}.post_attention_layernorm.weight"])
        m = f"layers.{L}.mlp."
        gate = np.minimum(y @ W[m + "gate_proj.weight"].T, LIMIT)
        up = np.clip(y @ W[m + "up_proj.weight"].T, -LIMIT, LIMIT)
        h = h + (gate / (1 + np.exp(-ALPHA * gate)) * (up + 1)) @ W[m + "down_proj.weight"].T
    return _gnorm(h, W["norm.weight"]), keep


def _drafter(W):
    from vmlx_engine.dspark.drafter import DSparkDrafter, is_minimax_config, minimax_config

    raw = _raw_config()
    assert is_minimax_config(raw)
    d = DSparkDrafter(minimax_config(raw))
    d.load_weights([(k, mx.array(v)) for k, v in W.items()], strict=True)
    return d


def test_minimax_block_forward_matches_reference_with_a_sliding_window():
    rng = np.random.default_rng(0)
    W = _weights(rng)
    feats = rng.standard_normal((CTX, TAPS * HID)).astype(np.float32)
    block = rng.standard_normal((BLOCK, HID)).astype(np.float32)
    ref, keep = _reference(W, feats, block)
    ctx_keep = keep[:, :CTX]
    assert (~ctx_keep).any() and (ctx_keep.sum(1) != ctx_keep.sum(1)[0]).any(), \
        "window excludes nothing or excludes the same keys for every row: the test would prove nothing"
    d = _drafter(W)
    d.add_context(mx.array(feats)[None])
    got = np.array(d(mx.array(block)[None]))[0]
    np.testing.assert_allclose(got, ref, rtol=0, atol=REL_TOL * np.abs(ref).max())
    assert d.ctx_len == CTX  # the block rows are not kept as context


def test_minimax_config_rejects_what_the_flavor_does_not_implement():
    import pytest
    from vmlx_engine.dspark.drafter import minimax_config

    raw = _raw_config()
    raw["text_config"]["layer_types"] = ["full_attention", "sliding_attention"]
    with pytest.raises(ValueError, match="non-sliding"):
        minimax_config(raw)

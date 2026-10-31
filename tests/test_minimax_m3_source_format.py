# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 official checkpoint (MXFP8 linears, NVFP4 experts) loads as stored.

A tiny model gets random stored bytes in both formats under the checkpoint's own
tensor names. Model.sanitize converts them and quantizes the modules to match,
and the result must equal a float model whose weights are dequantized here from
the format definitions:
  MXFP8: e4m3 weight, e8m0 scale per 32 inputs (``weight_scale_inv``)
  NVFP4: two e2m1 codes per byte (low nibble first), e4m3 scale per 16 inputs,
         weight divided by ``weight_global_scale``
A wrong nibble order, scale meaning or missing global scale changes the output
by O(1), far above the tolerance (which allows for TF32 matmuls on M5).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

HID, HEADS, KV, HD, INTER, E, VOCAB = 64, 4, 2, 16, 32, 4, 96
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def _e4m3(b):
    b = b.astype(np.int64)
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(s == 1, -v, v)


def _args():
    from vmlx_engine.models.minimax_m3.minimax_m3 import ModelArgs

    return ModelArgs(
        hidden_size=HID, num_hidden_layers=2, intermediate_size=INTER, dense_intermediate_size=HID,
        shared_intermediate_size=INTER, num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HD,
        rotary_dim=8, rope_theta=10_000.0, vocab_size=VOCAB, num_local_experts=E, num_experts_per_tok=2,
        moe_layer_freq=[0, 1], index_n_heads=KV, index_head_dim=HD, index_block_size=4, index_topk_blocks=2,
        sparse_attention_freq=[1, 1],
    )


def _mxfp8(rng, out_d, in_d):
    # realistic magnitudes: e4m3 exponent field 5..8 (0.25-3.75), e8m0 scale 2^-6..2^-4 -> |w| ~0.004-0.23
    sign = rng.integers(0, 2, (out_d, in_d)) << 7
    w = (sign | (rng.integers(5, 9, (out_d, in_d)) << 3) | rng.integers(0, 8, (out_d, in_d))).astype(np.uint8)
    s = rng.integers(121, 124, (out_d, in_d // 32)).astype(np.uint8)
    return w, s, _e4m3(w) * np.repeat(2.0 ** (s.astype(np.float64) - 127), 32, axis=1)


def _nvfp4(rng, out_d, in_d):
    packed = rng.integers(0, 256, (out_d, in_d // 2)).astype(np.uint8)
    scale = rng.integers(0x30, 0x48, (out_d, in_d // 16)).astype(np.uint8)  # positive e4m3 around 0.1-1
    g = float(rng.uniform(50, 200))
    codes = np.empty((out_d, in_d), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    return packed, scale, g, E2M1[codes] * np.repeat(_e4m3(scale), 16, axis=1) / g


def _checkpoint(seed=0):
    """(source tensors with checkpoint names, float weights with the model's names)"""
    rng = np.random.default_rng(seed)
    src, flt = {}, {}
    f32 = lambda *s: rng.standard_normal(s).astype(np.float32) * 0.2  # noqa: E731
    for name, v in (("model.embed_tokens.weight", f32(VOCAB, HID)), ("lm_head.weight", f32(VOCAB, HID)),
                    ("model.norm.weight", f32(HID))):
        src["language_model." + name if name.startswith("model.") else "language_model." + name] = v
        flt[name] = v

    def lin(src_name, model_name, out_d, in_d):
        w, s, ref = _mxfp8(rng, out_d, in_d)
        src[src_name + ".weight"], src[src_name + ".weight_scale_inv"] = w, s
        flt[model_name + ".weight"] = ref.astype(np.float32)

    for L in range(2):
        p, q = f"language_model.model.layers.{L}.", f"model.layers.{L}."
        for k in ("input_layernorm", "post_attention_layernorm"):
            src[p + k + ".weight"] = flt[q + k + ".weight"] = f32(HID)
        a = "self_attn."
        for k, o, i in (("q_proj", HEADS * HD, HID), ("k_proj", KV * HD, HID), ("v_proj", KV * HD, HID),
                        ("o_proj", HID, HEADS * HD)):
            lin(p + a + k, q + a + k, o, i)
        lin(p + a + "index_q_proj", q + a + "indexer.index_q_proj", KV * HD, HID)
        lin(p + a + "index_k_proj", q + a + "indexer.index_k_proj", HD, HID)
        for k, mk in (("q_norm", "q_norm"), ("k_norm", "k_norm"), ("index_q_norm", "indexer.index_q_norm"),
                      ("index_k_norm", "indexer.index_k_norm")):
            src[p + a + k + ".weight"] = flt[q + a + mk + ".weight"] = f32(HD)
        if L == 0:
            for k, o, i in (("gate_proj", HID, HID), ("up_proj", HID, HID), ("down_proj", HID, HID)):
                lin(p + "mlp." + k, q + "mlp." + k, o, i)
            continue
        m = p + "block_sparse_moe."
        src[m + "gate.weight"] = flt[q + "mlp.gate.weight"] = f32(E, HID)
        src[m + "e_score_correction_bias"] = flt[q + "mlp.e_score_correction_bias"] = f32(E)
        for k, o, i in (("gate_proj", INTER, HID), ("up_proj", INTER, HID), ("down_proj", HID, INTER)):
            lin(m + "shared_experts." + k, q + "mlp.shared_experts." + k, o, i)
        for wn, proj, o, i in (("w1", "gate_proj", INTER, HID), ("w3", "up_proj", INTER, HID),
                               ("w2", "down_proj", HID, INTER)):
            stack = []
            for e in range(E):
                packed, scale, g, ref = _nvfp4(rng, o, i)
                b = f"{m}experts.{e}.{wn}."
                src[b + "weight_packed"], src[b + "weight_scale"] = packed, scale
                src[b + "weight_global_scale"] = np.array(g, np.float32)
                src[b + "input_global_scale"] = np.array(1.0, np.float32)
                stack.append(ref)
            flt[q + f"mlp.switch_mlp.{proj}.weight"] = np.stack(stack).astype(np.float32)
    return src, flt


def _model(weights, convert):
    from vmlx_engine.models.minimax_m3.minimax_m3 import Model

    m = Model(_args())
    w = {k: mx.array(v) for k, v in weights.items()}
    m.load_weights(list((m.sanitize(w) if convert else w).items()), strict=True)
    mx.eval(m.parameters())
    return m


def test_source_checkpoint_loads_as_stored_and_matches_the_spec_dequant():
    from vmlx_engine.models.minimax_m3.source_format import NVFP4SwitchLinear, is_source_checkpoint

    src, flt = _checkpoint()
    assert is_source_checkpoint(src) and not is_source_checkpoint(flt)
    got_m, ref_m = _model(src, convert=True), _model(flt, convert=False)
    sw = got_m.model.layers[1].mlp.switch_mlp
    assert all(isinstance(getattr(sw, n), NVFP4SwitchLinear) for n in ("gate_proj", "up_proj", "down_proj"))
    assert got_m.model.layers[0].self_attn.q_proj.mode == "mxfp8"

    tokens = mx.array(np.random.default_rng(3).integers(0, VOCAB, (1, 13)))
    ref = np.array(ref_m(tokens))[0]
    got = np.array(got_m(tokens))[0]
    tol = 1e-2 * np.abs(ref).max()
    np.testing.assert_allclose(got, ref, rtol=0, atol=tol)

    # fail closed: the routed experts must matter, or expert-layout errors would pass unseen
    for n in ("gate_proj", "up_proj", "down_proj"):
        p = getattr(sw, n)
        p.inv_global_scale = p.inv_global_scale * 0
    no_experts = np.array(got_m(tokens))[0]
    assert np.abs(no_experts - ref).max() > 10 * tol, "routed experts barely change the output"


def test_jang_style_weights_do_not_trigger_the_source_path():
    from vmlx_engine.models.minimax_m3.source_format import is_source_checkpoint

    assert not is_source_checkpoint({"model.layers.0.self_attn.q_proj.weight": 0,
                                     "model.layers.0.self_attn.q_proj.scales": 0,
                                     "model.layers.0.self_attn.q_proj.biases": 0})

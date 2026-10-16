"""Nemotron-H puzzle blocks (per-block MoE widths) + JANGH gate-less relu2 experts (TQSwitchMLP).

Reference = the stock mlx_lm SwitchMLP math on the dequantized original-basis weights; the JANGH module must match it on
both the decode (gather qmv) and the prefill (sorted qmm) paths, at sizes that are not tile multiples."""
import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.jangh import contract as C
from vmlx_engine.jangh.format import codebook, dequant_rows, encode_rows, h32, pack_bitstream
from vmlx_engine.jangh.switch import SORT_THRESHOLD, TQSwitchMLP


def _encode(W: mx.array, bits: int):
    """(E, N, K) -> packed, fp16 scales, original-basis dequantized weights (hadamard32 rotation)."""
    E, N, K = W.shape
    Wr = h32(W.reshape(E * N, K))
    q, s = encode_rows(Wr, bits)
    s16 = s.astype(mx.float16)
    packed = pack_bitstream(q.astype(mx.uint32), bits).reshape(E, N, K * bits // 32)
    W_hat = h32(dequant_rows(q, s16.astype(mx.float32), bits)).reshape(E, N, K)
    return packed, s16.reshape(E, N), W_hat


def _ref(x, idx, W1, W2):
    """relu2 SwitchMLP reference, float32: (T, k, D)."""
    a = mx.einsum("td,tkid->tki", x.astype(mx.float32), W1[idx])
    a = mx.square(mx.maximum(a, 0))
    return mx.einsum("tki,tkdi->tkd", a, W2[idx])


@pytest.mark.parametrize("bits1,bits2", [(2, 2), (2, 3), (3, 4)])
def test_tqswitchmlp_matches_dense_reference(bits1, bits2):
    mx.random.seed(0)
    E, D, I, k = 7, 128, 192, 3                    # I = 192: not a multiple of the 64-row prefill tile pair
    W1 = mx.random.normal((E, I, D)) * 0.05
    W2 = mx.random.normal((E, D, I)) * 0.05
    p1, s1, W1h = _encode(W1, bits1)
    p2, s2, W2h = _encode(W2, bits2)
    m = TQSwitchMLP(D, I, E, bits1, bits2, rotation_fc1="hadamard32", rotation_fc2="hadamard32")
    m.fc1.tq2_packed, m.fc1.tq2_scales = p1, s1
    m.fc2.tq2_packed, m.fc2.tq2_scales = p2, s2
    for T in (1, 3, 45):                           # 3 / 9 routed pairs: decode; 45 x 3 = 135 >= 64: sorted prefill
        x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
        idx = mx.array(np.random.default_rng(T).integers(0, E, (T, k)).astype(np.uint32))
        y = m(x, idx).astype(mx.float32)
        r = _ref(x, idx, W1h, W2h)
        rel = float(mx.linalg.norm(y - r) / mx.linalg.norm(r))
        path = "sorted" if T * k >= SORT_THRESHOLD else "decode"
        assert y.shape == (T, k, D)
        assert rel < 2e-2, f"T={T} ({path}) bits {bits1}/{bits2}: rel err {rel:.4f}"


def _tiny_cfg():
    return {
        "model_type": "nemotron_h", "vocab_size": 64, "hidden_size": 128, "intermediate_size": 96,
        "max_position_embeddings": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "attention_bias": False, "mamba_num_heads": 4, "mamba_head_dim": 32, "mamba_proj_bias": False,
        "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2, "mlp_bias": False, "layer_norm_epsilon": 1e-5,
        "use_bias": False, "use_conv_bias": True, "moe_latent_size": 64, "moe_shared_expert_intermediate_size": 96,
        "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 5.0, "n_shared_experts": 1,
        "time_step_limit": [0.0, 1e30], "num_hidden_layers": 4,
        "block_configs": [{"block_type": "mamba"},
                          {"block_type": "moe", "n_routed_experts": 6, "moe_intermediate_size": 128, "num_experts_per_tok": 2},
                          {"block_type": "attention"},
                          {"block_type": "moe", "n_routed_experts": 8, "moe_intermediate_size": 64, "num_experts_per_tok": 3}],
    }


def test_puzzle_blocks_get_their_own_shapes_and_fp32_router():
    from vmlx_engine.utils.nemotron_puzzle import ensure_puzzle_support
    from mlx_lm.models import nemotron_h
    cfg = _tiny_cfg()
    assert ensure_puzzle_support(cfg)
    args = nemotron_h.ModelArgs.from_dict(cfg)
    model = nemotron_h.Model(args)
    l1, l3 = model.layers[1].mixer, model.layers[3].mixer
    assert l1.switch_mlp.fc1.weight.shape == (6, 128, 64) and l1.switch_mlp.fc2.weight.shape == (6, 64, 128)
    assert l3.switch_mlp.fc1.weight.shape == (8, 64, 64) and l3.gate.weight.shape == (8, 128)
    assert l1.num_experts_per_tok == 2 and l3.num_experts_per_tok == 3
    assert type(l1.gate).__name__ == "Float32Router"
    out = model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(out)
    assert out.shape == (1, 5, 64) and bool(mx.all(mx.isfinite(out)).item())
    # a model WITHOUT block_configs is built exactly as before (global widths, stock router)
    plain = dict(cfg); plain.pop("block_configs")
    plain.update({"n_routed_experts": 4, "moe_intermediate_size": 32, "num_experts_per_tok": 2,
                  "layers_block_type": ["mamba", "moe", "attention", "moe"]})
    pm = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(plain))
    assert pm.layers[3].mixer.switch_mlp.fc1.weight.shape == (4, 32, 64)
    assert type(pm.layers[1].mixer.gate).__name__ == "MoEGate"


def _jangtq_decl():
    books = {}
    for b in (2, 3, 4):
        a, be = C.CUBIC_PARAMS[b]
        books[str(b)] = {"alpha": a, "beta": be,
                         "levels": [float(np.float32((i - ((1 << b) - 1) / 2) * (a + be * ((i - ((1 << b) - 1) / 2) ** 2)))) for i in range(1 << b)]}
    return {"version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "codebook_family": "odd-cubic",
            "rotation": "hadamard32", "codebooks": books}


def test_install_replaces_switchmlp_and_fails_closed():
    from vmlx_engine.jangh.install import install_jangh
    from vmlx_engine.utils.nemotron_puzzle import ensure_puzzle_support
    from mlx_lm.models import nemotron_h
    cfg = _tiny_cfg(); ensure_puzzle_support(cfg)
    q = {}
    for L in (1, 3):
        q[f"backbone.layers.{L}.mixer.switch_mlp.fc1"] = {"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}
        q[f"backbone.layers.{L}.mixer.switch_mlp.fc2"] = {"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"}
    full = dict(cfg, jangtq=_jangtq_decl(), quantization=q)
    model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(full))
    assert install_jangh(model, full) == 2
    sw = model.layers[3].mixer.switch_mlp
    assert isinstance(sw, TQSwitchMLP) and sw.fc1.bits == 2 and sw.fc2.bits == 3
    assert sw.fc1.tq2_packed.shape == (8, 64, 64 * 2 // 32) and sw.fc2.tq2_scales.shape == (8, 64)
    broken = dict(full, quantization={k: v for k, v in q.items() if not k.endswith("3.mixer.switch_mlp.fc2")})
    model2 = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(broken))
    with pytest.raises(ValueError):
        install_jangh(model2, broken)


def test_uniform_nemotron_h_moe_gets_fp32_router():
    """Nemotron-3.5-Super-VL (no block_configs): the router logits are float32 like NVIDIA's NemotronHTopkRouter."""
    from vmlx_engine.utils.nemotron_puzzle import ensure_puzzle_support
    from mlx_lm.models import nemotron_h
    cfg = _tiny_cfg(); cfg.pop("block_configs")
    cfg.update({"n_routed_experts": 16, "moe_intermediate_size": 32, "num_experts_per_tok": 4,
                "layers_block_type": ["mamba", "moe", "attention", "moe"]})
    assert ensure_puzzle_support(cfg) is False                         # no per-block rebuild ...
    gate = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(cfg)).layers[1].mixer.gate
    gate.weight = (mx.random.normal(gate.weight.shape) * 0.2).astype(mx.bfloat16)
    gate.e_score_correction_bias = mx.random.normal((16,)) * 0.01
    x = mx.random.normal((7, cfg["hidden_size"])).astype(mx.bfloat16)
    inds, w = gate(x)                                                   # ... but the fp32 router
    ref_i, ref_w = nemotron_h.group_expert_select(x.astype(mx.float32) @ gate.weight.astype(mx.float32).T,
                                                  gate.e_score_correction_bias, 4, gate.n_group, gate.topk_group,
                                                  gate.routed_scaling_factor, gate.norm_topk_prob)
    assert bool(mx.array_equal(mx.sort(inds, axis=-1), mx.sort(ref_i, axis=-1)).item())
    assert float(mx.max(mx.abs(mx.sort(w, axis=-1) - mx.sort(ref_w, axis=-1))).item()) < 1e-6


def test_nemotron_h_mtp_expected_index_layers():
    from vmlx_engine.native_mtp import expected_indexed_mtp_layers
    assert expected_indexed_mtp_layers({"num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"]}, 1) == 2
    assert expected_indexed_mtp_layers({"mtp_hybrid_override_pattern": "*E"}, 1) == 2
    assert expected_indexed_mtp_layers({"num_nextn_predict_layers": 1}, 1) == 1          # one mtp.N per step elsewhere
    assert expected_indexed_mtp_layers({"text_config": {"num_nextn_predict_layers": 2}}, 2) == 2
    assert expected_indexed_mtp_layers({}, None) is None


def test_super_vl_bundle_uses_the_nemotron_vl_route(tmp_path):
    import json as _json
    from vmlx_engine.models.nemotron_bolt_vl import is_nemotron_bolt_vl_bundle
    (tmp_path / "config.json").write_text(_json.dumps({"model_type": "nemotron_h", "num_nextn_predict_layers": 1}))
    omni = {"vision_config": {"model_type": "radio"}, "img_context_token_id": 18, "llm_config": {"num_nextn_predict_layers": 1}}
    (tmp_path / "config_omni.json").write_text(_json.dumps(omni))
    assert is_nemotron_bolt_vl_bundle(tmp_path)                         # uniform MoE (no block_configs) qualifies
    (tmp_path / "config_omni.json").write_text(_json.dumps({**omni, "sound_config": {"model_type": "parakeet"}}))
    assert not is_nemotron_bolt_vl_bundle(tmp_path)                     # omni with a sound tower: not this route

"""MiMo-V2.6 JANGHT bundles: the text loader builds the stock v26 model, installs JTMixedSwitchGLU routed stacks from the
per-module contract (JANGT gate/up + JANGH down here), quantizes the non-experts affine and loads every tensor strictly.

Synthetic two-layer bundle (layer 0 dense full attention, layer 1 MoE SWA with sinks). Checked:
  * the bundle is recognised as a MiMo-V2.6 contract from metadata (format "jangtq2" + routed jangt contract);
  * every routed tensor and every non-expert tensor lands in the module that the contract names (byte equality);
  * the MoE forward equals JTMixedSwitchGLU.routed on the router's own selection (no stray patch, fp32 combine);
  * the full forward is finite and decode-with-cache == one-shot forward on the last position (cache wiring);
  * an incomplete routed contract fails closed.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.mimo_v2.v26_model")

from mlx.utils import tree_flatten  # noqa: E402

from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.models.mimo_v26_contract import read_mimo_v26_contract  # noqa: E402
from vmlx_engine.models.mimo_v26_janght import build_mimo_v26_janght_model, is_janght_bundle  # noqa: E402

D, I, E, TOPK, V = 1024, 256, 8, 4, 512
CFG = {
    "model_type": "mimo_v2", "vocab_size": V, "hidden_size": D, "intermediate_size": 512, "moe_intermediate_size": I,
    "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64, "v_head_dim": 64,
    "swa_num_attention_heads": 8, "swa_num_key_value_heads": 4, "swa_head_dim": 64, "swa_v_head_dim": 64,
    "layernorm_epsilon": 1e-6, "rope_theta": 10_000_000.0, "swa_rope_theta": 10_000.0, "partial_rotary_factor": 0.5,
    "sliding_window": 16, "attention_value_scale": 0.707, "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True, "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
    "n_routed_experts": E, "num_experts_per_tok": TOPK, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc", "tie_word_embeddings": False,
}
GU = {"mode": "jangt", "kbits": 2}
DN = {"mode": "jangtq2", "bits": 3, "rotation": "hadamard32"}


def _switch(seed=0):
    rng = np.random.default_rng(seed)
    m = SW.JTMixedSwitchGLU(D, I, E, GU, DN, 0, 0, 0.0)
    for p in (m.gate_proj, m.up_proj):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.002, 0.004, size=p.jt_scale.shape).astype(np.float32))
    su = mx.array(rng.choice([-1.0, 1.0], size=m.gate_proj.jt_su.shape).astype(np.float32))
    m.gate_proj.jt_su = su; m.up_proj.jt_su = su
    d = m.down_proj
    d.tq2_packed = mx.array(rng.integers(0, 2 ** 32, size=d.tq2_packed.shape, dtype=np.uint64).astype(np.uint32))
    d.tq2_scales = mx.array(rng.uniform(0.002, 0.004, size=d.tq2_scales.shape).astype(np.float16))
    return m


def _bundle(tmp_path, drop=None):
    from jang_tools.mimo_v2 import v26_model as T

    mx.random.seed(3)
    ref = T.Model(T.ModelArgs.from_dict(CFG))
    w = {}
    q = {"group_size": 64, "bits": 8, "mode": "affine"}
    for k, v in tree_flatten(ref.parameters()):
        if ".switch_mlp." in k:
            continue
        if k.endswith(".weight") and v.ndim == 2 and ".mlp.gate." not in k:
            qq, s, b = mx.quantize(v.astype(mx.bfloat16), group_size=64, bits=8)
            base = k[: -len(".weight")]
            w[base + ".weight"], w[base + ".scales"], w[base + ".biases"] = qq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        else:
            w[k] = v.astype(mx.float32) if (".mlp.gate." in k or "sink" in k) else v.astype(mx.bfloat16)
    sw = _switch()
    for k, v in tree_flatten(sw.parameters()):
        w[f"model.layers.1.mlp.switch_mlp.{k}"] = v
    for p, ent in (("gate_proj", GU), ("up_proj", GU), ("down_proj", DN)):
        if p != drop:
            q[f"model.layers.1.mlp.switch_mlp.{p}"] = ent
    cfg = dict(CFG, quantization=q, jangt={"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                                         "outlier_columns": {}})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "jang_config.json").write_text(json.dumps({"format": "jangtq2", "format_version": 2, "drop_mtp": True}))
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), w)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in w}}))
    return w, sw


def test_contract_and_strict_load(tmp_path):
    w, sw = _bundle(tmp_path)
    c = read_mimo_v26_contract(tmp_path)
    assert c is not None and is_janght_bundle(c)
    model = build_mimo_v26_janght_model(tmp_path)
    got = dict(tree_flatten(model.parameters()))
    assert set(got) == set(w), sorted(set(got) ^ set(w))[:10]
    for k, v in w.items():
        assert got[k].dtype == v.dtype and bool(mx.array_equal(got[k], v)), k
    assert isinstance(model.model.layers[1].mlp.switch_mlp, SW.JTMixedSwitchGLU)
    # MoE == routed() of the same stack on the router's own choice
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((1, 7, D)).astype(np.float32) * 0.5).astype(mx.bfloat16)
    moe = model.model.layers[1].mlp
    idx, wt = moe.gate(x)
    a = moe(x).astype(mx.float32)
    b = moe.switch_mlp.routed(x, idx, wt).astype(x.dtype).astype(mx.float32)
    assert bool(mx.array_equal(a, b))
    assert float(mx.abs(a).max().item()) > 0, "fixture must produce a non-zero expert output"


def test_forward_finite_and_cache_consistent(tmp_path):
    _bundle(tmp_path)
    model = build_mimo_v26_janght_model(tmp_path)
    ids = mx.array(np.random.default_rng(1).integers(0, V, size=(1, 24)).astype(np.int32))
    full = model(ids)[0].astype(mx.float32)
    assert bool(mx.all(mx.isfinite(full)).item())
    cache = model.make_cache()
    model(ids[:, :-1], cache=cache)
    last = model(ids[:, -1:], cache=cache)[0, -1].astype(mx.float32)
    rel = float((mx.linalg.norm(last - full[-1]) / mx.linalg.norm(full[-1])).item())
    assert rel < 3e-2, rel


def test_incomplete_contract_fails_closed(tmp_path):
    _bundle(tmp_path, drop="down_proj")
    with pytest.raises(ValueError):
        build_mimo_v26_janght_model(tmp_path)


def test_contract_needs_routed_jangt_metadata(tmp_path):
    _bundle(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    jc = tmp_path / "jang_config.json"
    jc.write_text(json.dumps({"weight_format": "janght"}))              # branded key alone is not the runtime contract
    assert read_mimo_v26_contract(tmp_path) is None
    jc.write_text(json.dumps({"format": "jangtq2"}))
    no_jangt = {k: v for k, v in cfg.items() if k != "jangt"}
    (tmp_path / "config.json").write_text(json.dumps(no_jangt))
    assert read_mimo_v26_contract(tmp_path) is None
    affine_routed = dict(cfg, quantization={k: ({"group_size": 64, "bits": 4} if ".mlp.switch_mlp." in k else v)
                                            for k, v in cfg["quantization"].items()})
    (tmp_path / "config.json").write_text(json.dumps(affine_routed))
    assert read_mimo_v26_contract(tmp_path) is None
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    c = read_mimo_v26_contract(tmp_path)
    assert c is not None and is_janght_bundle(c)


@pytest.mark.parametrize("rows, keys", [(1100, 2048), (200, 2048), (1, 2048)])
def test_full_attention_padded_fused_path_matches_native(monkeypatch, rows, keys):
    """MiMo's 192/128 head dims are outside MLX's fused attention; the long-context path pads to 256 (and lifts short
    chunks to the fused kernel's 1024-row floor with leading zero query rows). Same result as the native call."""
    import mlx.core as mx
    from vmlx_engine.models import mimo_v26_janght as J

    monkeypatch.setattr(J, "_SCORES_BUDGET_BYTES", 0)          # take the padded path at test sizes
    H, KV = 8, 2
    mx.random.seed(3)
    q = mx.random.normal((1, H, rows, 192)).astype(mx.bfloat16)
    k = mx.random.normal((1, KV, keys, 192)).astype(mx.bfloat16)
    v = mx.random.normal((1, KV, keys, 128)).astype(mx.bfloat16)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    mask = "causal" if rows > 1 else None
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=192 ** -0.5, mask=mask, sinks=sinks)
    out = J._full_attention(q, k, v, 192 ** -0.5, mask, sinks)
    assert out.shape == ref.shape == (1, H, rows, 128)
    err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert err < 2e-2, err


@pytest.mark.parametrize("rows", [3, 8, 16])
def test_few_row_grouped_attention_matches_native(rows):
    """DFlash verify rows over a long cache take the grouped (GQA folded) path; same result as the native call."""
    import mlx.core as mx
    from vmlx_engine.models import mimo_v26_janght as J

    H, KV, T = 16, 2, 4200
    mx.random.seed(5)
    q = mx.random.normal((1, H, rows, 192)).astype(mx.bfloat16)
    k = mx.random.normal((1, KV, T, 192)).astype(mx.bfloat16)
    v = mx.random.normal((1, KV, T, 128)).astype(mx.bfloat16)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=192 ** -0.5, mask="causal", sinks=sinks)
    out = J._full_attention(q, k, v, 192 ** -0.5, "causal", sinks)
    err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert out.shape == ref.shape and err < 1e-2, err

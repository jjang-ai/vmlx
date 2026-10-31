# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 JANGHT bundles: JANGT routed experts + a mixed-mode trunk load and run.

A tiny minimax_m3_vl bundle is written the way the M3.1 JANGHT assembler writes one (tensor names and per-module
``quantization`` keys = the LIVE module paths after Model.sanitize; a top-level ``jangt`` block; jang_config.json
format "jangtq2"):
  - routed experts: JANGT units (trellis words, row scales, input signs, outlier columns with -1 padding, down column
    scales), ``{"mode": "jangt", "kbits": K}``, two different widths per layer;
  - trunk linears mixed per module: MLX mxfp8 ``{"mode": "mxfp8", "group_size": 32, "bits": 8}`` and affine
    ``{"group_size": 64, "bits": 4|6|8}``; embed affine 8, lm_head affine 6; router gates float32, unquantized.
It is loaded three ways: in memory (Model + nn.quantize with the entries + load_weights(strict=True)), through
mlx_lm.utils.load_model(strict=True), and through vMLX's JANG loader (load_jang_model, the route a bundle with
jang_config.json takes). Each must give every module exactly its entry's mode / bits / group size, bind every
parameter from the bundle, and produce the same logits.

Reference: the same model whose routed experts are plain float SwitchGLU weights dequantized from the JANGT units into
the raw input basis (rotation, signs, outlier columns and column scales folded in), run through M3's own SwiGLU-OAI.
Prefill (37 tokens x top-4 = 148 rows: the sorted NAX prefill on M5, past the sparse-attention top-k blocks) and
decode steps through the cache are compared. Control: the reference with a clamped-SiLU expert activation must miss
the tolerance by a wide margin.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.jangt import v2l as VL  # noqa: E402
from vmlx_engine.models.minimax_m3 import minimax_m3 as M  # noqa: E402

D, I, SH, DENSE, E, TOPK, VOCAB = 256, 128, 128, 256, 8, 4, 512
HEADS, KVH, HD, LAYERS = 4, 2, 64, 3
ALPHA, LIMIT = 1.702, 7.0
WIDTHS = {1: (2, 2.5), 2: (1.5, 2)}             # MoE layer -> (gate/up K, down K)
GU_COLS = {1: [3, 100, 201], 2: [17, 250]}      # layer-wide gate/up outlier columns

# per-module trunk quantization, keyed by the live module path (what the assembler writes)
MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}
A = lambda b: {"group_size": 64, "bits": b}  # noqa: E731
TRUNK = {
    "model.embed_tokens": A(8), "lm_head": A(6),
    "model.layers.0.self_attn.q_proj": MX8, "model.layers.0.self_attn.o_proj": MX8,
    "model.layers.0.self_attn.k_proj": A(6), "model.layers.0.self_attn.v_proj": A(6),
    "model.layers.0.mlp.gate_proj": MX8, "model.layers.0.mlp.up_proj": MX8, "model.layers.0.mlp.down_proj": MX8,
    "model.layers.1.self_attn.q_proj": A(4), "model.layers.1.self_attn.k_proj": A(4),
    "model.layers.1.self_attn.v_proj": A(4), "model.layers.1.self_attn.o_proj": A(4),
    "model.layers.1.self_attn.indexer.index_q_proj": MX8, "model.layers.1.self_attn.indexer.index_k_proj": MX8,
    "model.layers.1.mlp.shared_experts.gate_proj": A(4), "model.layers.1.mlp.shared_experts.up_proj": A(4),
    "model.layers.1.mlp.shared_experts.down_proj": A(8),
    "model.layers.2.self_attn.q_proj": MX8, "model.layers.2.self_attn.k_proj": MX8,
    "model.layers.2.self_attn.v_proj": MX8, "model.layers.2.self_attn.o_proj": MX8,
    "model.layers.2.self_attn.indexer.index_q_proj": A(8), "model.layers.2.self_attn.indexer.index_k_proj": A(8),
    "model.layers.2.mlp.shared_experts.gate_proj": MX8, "model.layers.2.mlp.shared_experts.up_proj": MX8,
    "model.layers.2.mlp.shared_experts.down_proj": MX8,
}


def _config(jangt: bool) -> dict:
    tc = {
        "model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": LAYERS, "intermediate_size": I,
        "dense_intermediate_size": DENSE, "shared_intermediate_size": SH, "num_attention_heads": HEADS,
        "num_key_value_heads": KVH, "head_dim": HD, "rotary_dim": 32, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB, "num_local_experts": E, "num_experts_per_tok": TOPK, "routed_scaling_factor": 2.0,
        "norm_topk_prob": True, "swiglu_alpha": ALPHA, "swiglu_limit": LIMIT, "moe_layer_freq": [0, 1, 1],
        "sparse_attention_config": {"sparse_num_index_heads": KVH, "sparse_index_dim": HD, "sparse_block_size": 8,
                                    "sparse_topk_blocks": 2, "sparse_local_block": 1,
                                    "sparse_attention_freq": [0, 1, 1]},
    }
    cfg = {"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
           "text_config": tc, "quantization": {"group_size": 64, "bits": 8, **TRUNK}}
    if jangt:
        outl = {}
        for L, (kg, kd) in WIDTHS.items():
            key = f"model.layers.{L}.mlp.switch_mlp"
            cfg["quantization"].update({f"{key}.gate_proj": {"mode": "jangt", "kbits": kg},
                                        f"{key}.up_proj": {"mode": "jangt", "kbits": kg},
                                        f"{key}.down_proj": {"mode": "jangt", "kbits": kd}})
            outl.update({f"{key}.gate_proj": len(GU_COLS[L]), f"{key}.up_proj": len(GU_COLS[L]), f"{key}.down_proj": 2})
        cfg["jangt"] = {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                        "outlier_columns": outl}
    return cfg


# ------------------------------------------------------------------------------------------------ weights
def _sylvester(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def _hblk(n_in):
    """Block-diagonal normalized Hadamard-128 (n_in x n_in): the runtime's rotate_rows as a matrix (symmetric)."""
    H = np.zeros((n_in, n_in))
    for b in range(0, n_in, 128):
        H[b:b + 128, b:b + 128] = _sylvester(128)
    return H


def _jt_unit(rng, n_in, n_out, K, scale, cols, col_scales, su=None):
    """Random JANGT projection tensors + its raw-basis float64 weights (E, n_out, n_in): rotated rows x H128 x
    (signs, 0 on the outlier columns, x per-expert column scales), plus the outlier column weights."""
    W = VL.words_per_row(n_in, 12, K)
    packed = rng.integers(0, 2 ** 32, size=(E, n_out, W), dtype=np.uint64).astype(np.uint32)
    sc = rng.uniform(*scale, size=(E, n_out)).astype(np.float32)
    if su is None:
        su = rng.choice([-1.0, 1.0], size=n_in).astype(np.float32)
    oc = np.asarray(cols, np.int32)
    wo = (rng.standard_normal((E, n_out, oc.shape[1])) * 0.05 * (oc >= 0)[:, None, :]).astype(np.float16)
    cs = rng.uniform(0.5, 1.5, size=(E, n_in)).astype(np.float32) if col_scales else None
    t = {"jt_packed": packed, "jt_scale": sc, "jt_su": su, "jt_ocol": oc, "jt_wout": wo}
    if cs is not None:
        t["jt_cs"] = cs
    Hb, raw = _hblk(n_in), np.zeros((E, n_out, n_in))
    for e in range(E):
        st = VL.unpack_states(packed[e], n_in, 12, K)
        w_rot = np.array(VL.values_of(mx.array(st), 1.0, 12, K)).astype(np.float64) * sc[e][:, None]
        keep = np.ones(n_in)
        keep[oc[e][oc[e] >= 0]] = 0.0
        s = su.astype(np.float64) * keep * (cs[e] if cs is not None else 1.0)
        with np.errstate(all="ignore"):          # Accelerate matmul raises spurious FP flags; checked finite below
            raw[e] = (w_rot @ Hb) * s[None, :]
        for j, c in enumerate(oc[e]):
            if c >= 0:
                raw[e][:, c] += wo[e, :, j].astype(np.float64)
    assert np.isfinite(raw).all()
    return t, raw


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """(bundle dir, reference model, in-memory JANGT model, bundle tensors)."""
    rng = np.random.default_rng(0)
    args = M.ModelArgs.from_dict(_config(jangt=False))
    ref = M.Model(args)
    # float weights: bf16 linears / embeddings, Gemma norms, float32 router + correction bias
    flat = {}
    for k, v in nn.utils.tree_flatten(ref.parameters()):
        if k.endswith("norm.weight"):
            flat[k] = mx.array(rng.standard_normal(v.shape).astype(np.float32) * 0.1).astype(mx.bfloat16)
        elif k.endswith(".mlp.gate.weight") or k.endswith("e_score_correction_bias"):
            flat[k] = mx.array(rng.standard_normal(v.shape).astype(np.float32) * (0.3 if "gate" in k else 0.05))
        elif "switch_mlp" in k:
            continue
        else:
            std = 1.0 if "embed_tokens" in k else 0.06
            flat[k] = mx.array(rng.standard_normal(v.shape).astype(np.float32) * std).astype(mx.bfloat16)
    ref.load_weights(list(flat.items()), strict=False)
    nn.quantize(ref, class_predicate=lambda p, m: TRUNK.get(p, False))      # the trunk exactly as the bundle stores it
    tensors = {k: v for k, v in nn.utils.tree_flatten(ref.parameters()) if "switch_mlp" not in k}
    for L, (kg, kd) in WIDTHS.items():
        gu_cols = np.tile(np.array(GU_COLS[L]), (E, 1))
        dn_cols = np.stack([rng.choice(I, size=2, replace=False) for _ in range(E)])
        dn_cols[1, 1] = -1                                                   # padding slot
        su_gu = rng.choice([-1.0, 1.0], size=D).astype(np.float32)         # gate and up share the layer's signs
        sw = ref.model.layers[L].mlp.switch_mlp
        for p, n_in, n_out, K, scale, cols, cs, su in (("gate", D, I, kg, (0.15, 0.35), gu_cols, False, su_gu),
                                                       ("up", D, I, kg, (0.15, 0.35), gu_cols, False, su_gu),
                                                       ("down", I, D, kd, (0.01, 0.02), dn_cols, True, None)):
            t, raw = _jt_unit(rng, n_in, n_out, K, scale, cols, cs, su)
            for leaf, v in t.items():
                tensors[f"model.layers.{L}.mlp.switch_mlp.{p}_proj.{leaf}"] = mx.array(v)
            getattr(sw, f"{p}_proj").weight = mx.array(raw.astype(np.float32))
    cfg = _config(jangt=True)
    d = tmp_path_factory.mktemp("m3_janght")
    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), tensors, metadata={"format": "mlx"})
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    jcfg = {"format": "jangtq2", "format_version": 2, "format_name": "JANGHT", "model_name": "tiny-m3-janght"}
    (d / "jang_config.json").write_text(json.dumps(jcfg))
    # the in-memory JANGT model: install (Model.__init__), quantize with the entries, strict load
    mem = M.Model(M.ModelArgs.from_dict(cfg))
    nn.quantize(mem, class_predicate=lambda p, m: cfg["quantization"].get(p, False))
    mem.load_weights(list(tensors.items()), strict=True)
    mx.eval(ref.parameters(), mem.parameters())
    return d, ref, mem, tensors, cfg


# ------------------------------------------------------------------------------------------------ checks
def _module_quant(model) -> dict:
    """Live path -> (kind, mode, bits, group_size | kbits) for every quantized module."""
    out = {}
    for path, m in model.named_modules():
        if isinstance(m, SW.JTSwitchLinear):
            out[path] = ("jangt", m.kbits)
        elif isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            out[path] = ("mlx", m.mode, m.bits, m.group_size)
    return out


def _expected_quant(cfg) -> dict:
    exp = {}
    for k, v in cfg["quantization"].items():
        if not isinstance(v, dict):
            continue
        exp[k] = ("jangt", v["kbits"]) if v.get("mode") == "jangt" else ("mlx", v.get("mode", "affine"), v["bits"],
                                                                          v["group_size"])
    return exp


def _assert_loaded(model, tensors, cfg):
    assert _module_quant(model) == _expected_quant(cfg)                 # every entry landed, nothing else quantized
    params = dict(nn.utils.tree_flatten(model.parameters()))
    assert set(params) == set(tensors)                                  # every parameter bound, no stray tensor
    for k, v in tensors.items():
        assert params[k].dtype == v.dtype and mx.array_equal(params[k], v).item(), k
    for L in WIDTHS:
        sw = model.model.layers[L].mlp.switch_mlp
        assert isinstance(sw, SW.JTMixedSwitchGLU) and (sw.act, sw.alpha, sw.limit) == ("oai", ALPHA, LIMIT)
    assert isinstance(model.model.layers[1].mlp.gate, nn.Linear)        # router stays float
    assert model.model.layers[1].mlp.gate.weight.dtype == mx.float32


TOKENS = np.random.default_rng(7).integers(0, VOCAB, size=40)


def _run(model, n_prefill=37):
    """Teacher-forced logits: one 37-token prefill (148 routed rows), then 3 one-token decode steps on the cache."""
    cache = model.make_cache()
    outs = [model(mx.array(TOKENS[None, :n_prefill]), cache=cache)[0].astype(mx.float32)]
    for t in range(n_prefill, len(TOKENS)):
        outs.append(model(mx.array(TOKENS[None, t:t + 1]), cache=cache)[0].astype(mx.float32))
    out = mx.concatenate(outs, axis=0)
    mx.eval(out)
    return np.array(out)


def _rel_rows(a, b):
    """Relative L2 error of every position's logits."""
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.abs(b).max() > 0
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


# Per-position logits error vs the dequantized-expert reference, measured on M5 Max (2026-10-27): median 0.99 %
# (max 13.6 % at one position). The noise floor of this tiny random model is the same: the reference against itself
# with its expert weights rounded to bf16 gives median 0.97 % (max 35 %). A single position can jump because a
# rounding-level change flips a discrete choice (router top-k or MSA block top-k), so the test bounds the median and
# the share of positions. The clamped-SiLU control: median 33 %, min 24 %.
TOL = 3e-2


def test_in_memory_model_matches_the_dequantized_expert_reference(bundle):
    d, ref, mem, tensors, cfg = bundle
    _assert_loaded(mem, tensors, cfg)
    got, want = _run(mem), _run(ref)
    err = _rel_rows(got, want)
    assert np.median(err[:37]) < TOL and np.median(err[37:]) < TOL, err          # prefill, decode steps
    assert np.mean(err < TOL) >= 0.9, err
    # control: the same reference with a clamped-SiLU expert activation
    for L in WIDTHS:
        ref.model.layers[L].mlp.switch_mlp.activation = _ClampedSilu()
    try:
        ctl = _run(ref)
    finally:
        for L in WIDTHS:
            ref.model.layers[L].mlp.switch_mlp.activation = M.SwiGLU(ALPHA, LIMIT)
    err_ctl = _rel_rows(got, ctl)
    assert np.median(err_ctl[:37]) > 5 * TOL and np.median(err_ctl[37:]) > 5 * TOL, err_ctl
    assert np.mean(err_ctl < TOL) == 0.0, err_ctl


class _ClampedSilu(nn.Module):
    def __call__(self, x_up, x_gate):
        return nn.silu(mx.minimum(x_gate, LIMIT)) * mx.clip(x_up, -LIMIT, LIMIT)


def test_mlx_lm_load_model_strict(bundle):
    from mlx_lm.utils import load_model

    from vmlx_engine.models.minimax_m3.minimax_m3_register import register_minimax_m3_runtime

    d, ref, mem, tensors, cfg = bundle
    register_minimax_m3_runtime()
    model, _ = load_model(d, strict=True)
    _assert_loaded(model, tensors, cfg)
    assert np.array_equal(_run(model), _run(mem))


def test_vmlx_jang_loader(bundle, monkeypatch):
    """load_jang_model: jang_config.json format jangtq2 -> the JANG v2 text loader (the M3.1 JANGHT bundle route)."""
    import mlx_lm.utils

    from vmlx_engine.utils import jang_loader

    d, ref, mem, tensors, cfg = bundle
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: object())       # no tokenizer in the bundle
    routes = []
    real_v2 = jang_loader._load_jang_v2
    monkeypatch.setattr(jang_loader, "_load_jang_v2", lambda *a, **k: routes.append("v2") or real_v2(*a, **k))
    model, _ = jang_loader.load_jang_model(d)
    assert routes == ["v2"]
    _assert_loaded(model, tensors, cfg)
    assert np.array_equal(_run(model), _run(mem))


def test_source_form_keys_do_not_reach_the_modules(bundle, tmp_path):
    """Documents the key contract: an entry keyed by the checkpoint name (language_model.model.*, block_sparse_moe,
    self_attn.index_*) is not the live module path, so mlx-lm's quantize predicate never sees it."""
    cfg = _config(jangt=True)
    live = set(_module_quant(bundle[2]))
    for k in cfg["quantization"]:
        if isinstance(cfg["quantization"][k], dict):
            assert k in live, k
    src_form = "language_model.model.layers.2.self_attn.index_q_proj"
    assert src_form not in live and "model.layers.2.self_attn.indexer.index_q_proj" in live

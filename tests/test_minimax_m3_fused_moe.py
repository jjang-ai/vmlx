# SPDX-License-Identifier: Apache-2.0
"""Fused MiniMax-M3 MoE block (vmlx_engine/models/minimax_m3/fused_moe.py, VMLX_M3_FUSED_MOE=1).

K6 m3_router (fp32 gemv + sigmoid, routed-input rotation), K7 m3_moe_gate_up (routing, JANGT gate / up per slot,
shared affine-4 gate / up), K8 m3_moe_down (JANGT down + combine, shared down, residual add). Checked:
  * each kernel's outputs and the whole block equal the unfused ops BIT FOR BIT (router logits through MLX's one-row
    gemv, compiled group_expert_select, JANGT v3 kernels, MLX one-row qmv, compiled SwiGLU, bf16 adds) at 1..16 rows,
    JANGT widths 1 / 1.5 / 2 / 2.5 / 3, outlier columns on and off, distinct / shared / identical routing;
  * row invariance: a row computed alone equals the same row inside a 16-row window;
  * the DSpark target with the fused block: every verify row == its decode step (both on the fused kernels), and the
    fused decode == the unfused decode;
  * controls: a perturbed per-row combine in K8, and the fast-exp sigmoid in K6, are caught.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.jangt import moe_kernels as MK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_moe as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import minimax_m3 as M  # noqa: E402

E = 16


def _jangt(D, I, Kgu, Kdn, nog, nod, rng):
    sw = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": Kgu}, {"mode": "jangt", "kbits": Kdn}, nog, nod,
                             7.0, "oai", 1.702)
    for p, n_in, n_out in ((sw.gate_proj, D, nog), (sw.up_proj, D, nog), (sw.down_proj, I, nod)):
        p.jt_packed = mx.array(rng.integers(0, 2 ** 32, size=p.jt_packed.shape, dtype=np.uint64).astype(np.uint32))
        p.jt_scale = mx.array(rng.uniform(0.02, 0.05, size=p.jt_scale.shape).astype(np.float32))
        p.jt_su = mx.array(rng.choice([-1.0, 1.0], size=p.jt_su.shape).astype(np.float32))
        if n_out:
            if p is sw.down_proj:
                cols = np.stack([rng.choice(n_in, n_out, replace=False) for _ in range(E)])
                cols[2, -1] = -1                                       # a padded entry
            else:
                cols = np.tile(rng.choice(n_in, n_out, replace=False), (E, 1))
            p.jt_ocol = mx.array(cols.astype(np.int32))
            p.jt_wout = mx.array((rng.standard_normal(p.jt_wout.shape) * 0.05).astype(np.float16))
        if p is sw.down_proj:
            p.jt_cs = mx.array(rng.uniform(0.5, 1.5, size=(E, n_in)).astype(np.float32))
    sw.up_proj.jt_su = sw.gate_proj.jt_su
    sw.up_proj.jt_ocol = sw.gate_proj.jt_ocol
    return sw


def _block(D=1024, I=512, Kgu=1.5, Kdn=2.5, nog=1, nod=3, seed=0, hot=None):
    """A SparseMoeBlock as M3.1 runs it: fp32 router + correction bias, JANGT routed experts (SwiGLU-OAI), affine-4
    group-64 shared expert. ``hot``: bias the router so every row picks these experts."""
    rng = np.random.default_rng(seed)
    args = M.ModelArgs(hidden_size=D, intermediate_size=I, shared_intermediate_size=I, num_local_experts=E,
                       num_experts_per_tok=4)
    b = M.SparseMoeBlock(args)
    b.switch_mlp = _jangt(D, I, Kgu, Kdn, nog, nod, rng)
    b.gate.weight = mx.array((rng.standard_normal((E, D)) * 0.05).astype(np.float32))
    bias = rng.standard_normal(E) * 0.05
    if hot is not None:
        bias[list(hot)] += 50.0
    b.e_score_correction_bias = mx.array(bias.astype(np.float32))
    for n in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(b.shared_experts, n)
        lin.weight = mx.array((rng.standard_normal(lin.weight.shape) * 1.5 / np.sqrt(lin.weight.shape[1]))
                              .astype(np.float32)).astype(mx.bfloat16)
        setattr(b.shared_experts, n, nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4))
    mx.eval(b.parameters())
    return b


def _rows(T, D, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = mx.array((rng.standard_normal((1, T, D)) * scale).astype(np.float32)).astype(mx.bfloat16)
    res = mx.array(rng.standard_normal((1, T, D)).astype(np.float32)).astype(mx.bfloat16)
    return x, res


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("Kgu,Kdn,nog,nod", [(1, 1.5, 2, 8), (1.5, 2.5, 1, 3), (2, 2, 0, 0), (2.5, 3, 0, 5),
                                             (3, 1, 1, 0)])
def test_fused_block_equals_unfused_bit_for_bit(Kgu, Kdn, nog, nod):
    b = _block(Kgu=Kgu, Kdn=Kdn, nog=nog, nod=nod, seed=int(Kgu * 10 + Kdn))
    for T in (1, 2, 3, 7, 16):
        x, res = _rows(T, 1024, seed=T)
        assert F.supported(b, x)
        ref = F.unfused_moe_residual(b, x, res)
        got = F.fused_moe_residual(b, x, res)
        assert float(mx.abs(ref.astype(mx.float32) - res.astype(mx.float32)).max()) > 0     # the MoE did something
        assert _eq(got, ref), (T, float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()))


def test_each_kernel_equals_its_unfused_ops():
    """K6 / K7 / K8 piece by piece against the ops they replace (router logits / sigmoid / routing, routed gate-up,
    shared SwiGLU, routed down + combine)."""
    from mlx_lm.models.deepseek_v3 import group_expert_select

    b = _block(Kgu=1.5, Kdn=2.5, nog=1, nod=3, seed=11)
    sw, sh = b.switch_mlp, b.shared_experts
    U = sw._unit()
    T = 9
    x, res = _rows(T, 1024, seed=5)
    x2 = x[0]
    sig, xr = F.router(x2, b.gate.weight, U["su_gu"], T)
    one = lambda f: mx.concatenate([f(x[:, r:r + 1]) for r in range(T)], axis=1)[0]  # noqa: E731
    assert _eq(sig, mx.sigmoid(one(b.gate)))
    assert _eq(xr, MK.rotate_rows(x2, U["su_gu"]))
    sel = [group_expert_select(b.gate(x[:, r:r + 1]), b.e_score_correction_bias, 4, 1, 1, 2.0, True) for r in range(T)]
    ref_inds = mx.concatenate([s[0] for s in sel], axis=1)[0]
    ref_scores = mx.concatenate([s[1] for s in sel], axis=1)[0]
    h, hs, inds, scores = F.gate_up(b, x2, sig, xr)
    assert _eq(inds, ref_inds.astype(mx.uint32)) and _eq(scores, ref_scores)
    assert _eq(h, MK.gather_gate_up_swiglu_v3(xr, x2, ref_inds.astype(mx.int32), U, out_dtype=mx.float32, limit=7.0,
                                              act="oai", alpha=1.702))
    assert _eq(hs, one(lambda r: M.swiglu(sh.up_proj(r), sh.gate_proj(r), 1.702, 7.0)))


@pytest.mark.parametrize("mode", ["distinct", "same"])
def test_rows_are_independent_of_the_window(mode):
    """Row r alone (a decode step) == row r inside a 16-row window, including when every row picks the same
    experts (``same``: the router bias forces experts 1, 5, 9, 14 on all rows)."""
    b = _block(Kgu=2, Kdn=2.5, nog=2, nod=8, seed=3, hot=(1, 5, 9, 14) if mode == "same" else None)
    x, res = _rows(16, 1024, seed=9)
    full = F.fused_moe_residual(b, x, res)
    for r in (0, 6, 15):
        alone = F.fused_moe_residual(b, x[:, r:r + 1], res[:, r:r + 1])
        assert _eq(alone, full[:, r:r + 1]), r
    if mode == "same":
        sel = M.group_expert_select(b.gate(x[:, :1]), b.e_score_correction_bias, 4, 1, 1, 2.0, True)[0]
        assert sorted(np.array(sel).reshape(-1).tolist()) == [1, 5, 9, 14]


def test_wider_and_other_shapes():
    """D 1536 / I 1024 (gemv tail-free, 3 / 2 blocks of 512), 4 and 13 rows."""
    b = _block(D=1536, I=1024, Kgu=1, Kdn=2, nog=0, nod=4, seed=21)
    for T in (4, 13):
        x, res = _rows(T, 1536, seed=T + 40)
        assert _eq(F.fused_moe_residual(b, x, res), F.unfused_moe_residual(b, x, res)), T


def test_unsupported_calls_decline():
    b = _block(seed=2)
    x, _ = _rows(17, 1024, seed=1)
    assert not F.supported(b, x)                                    # > 16 rows: prefill keeps the unfused path
    assert not F.supported(b, x[:, :2].astype(mx.float32))          # fp32 activations
    b2 = _block(seed=2)
    b2.shared_experts.down_proj = nn.QuantizedLinear.from_linear(
        nn.Linear(512, 1024, bias=False), group_size=64, bits=8)    # 8-bit shared down
    assert not F.supported(b2, x[:, :2])


# ----------------------------------------------------------------------------------------------- controls
def _patched_source(monkeypatch, name, old, new):
    real = F.metal_source

    def patched(n):
        header, body = real(n)
        if n == name:
            assert old in body or old in header
            return header.replace(old, new), body.replace(old, new)
        return header, body

    monkeypatch.setattr(F, "metal_source", patched)
    F._kernel.cache_clear()


def test_control_perturbed_combine_is_caught(monkeypatch):
    """Row 1's routed combine perturbed by 0.1 % (bf16 output: a smaller change can round away at the block's
    output, which is why test_each_kernel_equals_its_unfused_ops checks the fp32 intermediates exactly): the block
    equality check fails on row 1 only."""
    b = _block(seed=4)
    x, res = _rows(4, 1024, seed=4)
    ref = F.unfused_moe_residual(b, x, res)
    _patched_source(monkeypatch, "m3_moe_down", "acc[r] = fma(wgt, y, acc[r]);",
                    "acc[r] = t == 1u ? fma(wgt * 1.001f, y, acc[r]) : fma(wgt, y, acc[r]);")
    try:
        got = F.fused_moe_residual(b, x, res)
        assert not _eq(got[:, 1], ref[:, 1]) and _eq(got[:, 0], ref[:, 0])
    finally:
        F._kernel.cache_clear()


def test_control_fast_exp_sigmoid_is_caught(monkeypatch):
    """K6 with the JIT (fast) exp in the router sigmoid differs from group_expert_select's sigmoid."""
    b = _block(seed=6)
    sw = b.switch_mlp
    x, _ = _rows(16, 1024, seed=6, scale=3.0)
    ref = mx.sigmoid(mx.concatenate([b.gate(x[:, r:r + 1]) for r in range(16)], axis=1)[0])
    _patched_source(monkeypatch, "m3_router", "metal::precise::exp(", "metal::exp(")
    try:
        sig, _ = F.router(x[0], b.gate.weight, sw._unit()["su_gu"], 16)
        assert not _eq(sig, ref)
    finally:
        F._kernel.cache_clear()


# ----------------------------------------------------------------------------------------------- DSpark target
def _model(seed=0):
    """A 3-layer M3.1-shaped model the fused block covers: dense layer 0, MoE layers 1-2 (JANGT routed experts,
    affine-4 shared expert, fp32 router), affine-4 trunk, sparse attention with small blocks."""
    from mlx.utils import tree_flatten

    D, I = 1024, 512
    tc = {"model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": 3, "intermediate_size": I,
          "dense_intermediate_size": 1024, "shared_intermediate_size": I, "num_attention_heads": 8,
          "num_key_value_heads": 2, "head_dim": 64, "rotary_dim": 32, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
          "vocab_size": 512, "num_local_experts": E, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
          "norm_topk_prob": True, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [0, 1, 1],
          "sparse_attention_config": {"sparse_num_index_heads": 2, "sparse_index_dim": 64, "sparse_block_size": 16,
                                      "sparse_topk_blocks": 3, "sparse_local_block": 1,
                                      "sparse_attention_freq": [1, 1, 1]}}
    model = M.Model(M.ModelArgs.from_dict({"model_type": "minimax_m3_vl", "text_config": tc}))
    rng = np.random.default_rng(seed)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if "switch_mlp" in k:
            continue
        if k.endswith("norm.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.1, mx.bfloat16
        elif k.endswith("e_score_correction_bias"):
            a, dt = rng.standard_normal(v.shape) * 0.05, mx.float32
        elif k.endswith(".mlp.gate.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.05, mx.float32
        elif "embed_tokens" in k:
            a, dt = rng.standard_normal(v.shape), mx.bfloat16
        else:
            a, dt = rng.standard_normal(v.shape) * 1.5 / np.sqrt(v.shape[-1]), mx.bfloat16
        weights.append((k, mx.array(a.astype(np.float32)).astype(dt)))
    model.load_weights(weights, strict=False)
    for li in (1, 2):
        model.model.layers[li].mlp.switch_mlp = _jangt(D, I, (1.5, 2)[li - 1], (2.5, 1.5)[li - 1], li, 4, rng)

    def plan(path, module):
        if path.endswith("mlp.gate") or "switch_mlp" in path:
            return False
        return isinstance(module, (nn.Linear, nn.Embedding)) and {"group_size": 64, "bits": 4}

    nn.quantize(model, class_predicate=plan)
    mx.eval(model.parameters())
    return model


def test_dspark_verify_with_fused_moe_equals_decode_steps(monkeypatch):
    """VMLX_M3_FUSED_MOE=1: the verify takes every row through the fused kernels at once; each row == its decode step
    (also fused), and the fused decode == the unfused decode (the block is bit-identical)."""
    from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget

    tgt = MiniMaxM3DSparkTarget(_model())
    prompt = mx.array([np.random.default_rng(1).integers(0, 511, 77).tolist()])
    for width in (1, 5, 16):
        rows = mx.array([np.random.default_rng(width).integers(0, 511, width).tolist()])
        monkeypatch.setenv(F.ENV, "1")
        tgt.verify_paths = {k: 0 for k in tgt.verify_paths}
        cache = tgt.make_cache()
        tgt.prefill(prompt, cache, (), 32)
        logits, _, _ = tgt.verify(rows, cache, ())
        assert tgt.verify_paths.get("moe_fused", 0) == 2 and tgt.verify_paths["routed_rows"] == 0
        ref = tgt.make_cache()
        tgt.prefill(prompt, ref, (), 32)
        fused_steps = [tgt.decode(rows[0, r:r + 1], ref) for r in range(width)]
        monkeypatch.setenv(F.ENV, "0")
        plain = tgt.make_cache()
        tgt.prefill(prompt, plain, (), 32)
        plain_steps = [tgt.decode(rows[0, r:r + 1], plain) for r in range(width)]
        assert all(_eq(logits[r], fused_steps[r]) for r in range(width)), width
        assert all(_eq(fused_steps[r], plain_steps[r]) for r in range(width)), width

"""qwen4_exp JANGHT wiring: routed experts as JANGT trellis units (Qwen3.8-Flash-Next, 2026-10-16).

A tiny qwen4_exp tree (two backbone SparseMoeBlocks + one MTP SparseMoeBlock, the real class at the real routed
geometry D = 2560, I = 640, top-10 of 16 experts) goes through the loader's own steps:
  _install_routed_expert_formats -> _normalize_runtime_weight_names -> normalize_jangt_payload (encoder STAGE layout:
  ragged jt_packed + jt_off + jt_kbits) -> _normalize_jang_runtime_compute_dtypes -> _quantize_model -> strict
  load_weights,
with mixed widths (layer 0: gate+up K=1.5, down K=2; layer 1: gate+up K=1, down K=2.5), then the block output
(routed + gated shared expert) is checked against a dense float64 reference built from the dequantized stage tensors
(tests/jangt_lowbit_support.py; no Metal kernel in the reference), on the decode route and on the sorted prefill
route. MTP experts, routers and shared experts must stay as they were.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from vmlx_engine.jangh import kernels as JK  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.jangt import v2l as VL  # noqa: E402
from vmlx_engine.models.qwen4_exp import language as QL  # noqa: E402
from vmlx_engine.models.qwen4_exp import loader as QLD  # noqa: E402

from tests import jangt_lowbit_support as S  # noqa: E402

D, I, E, TOPK, SHARED = 2560, 640, 16, 10, 128
PLAN = {0: (1.5, 2), 1: (1, 2.5)}            # layer: (gate+up K, down K)
N_GU, N_DN = 3, 2
REF = S.load_encoder_oracle() or VL


class _Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = QL.SparseMoeBlock(args)


class _Stack(nn.Module):
    def __init__(self, args, n):
        super().__init__()
        self.layers = [_Layer(args) for _ in range(n)]


class _LM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = _Stack(args, len(PLAN))
        self.mtp = _Stack(args, 1)


class _Root(nn.Module):
    """language_model.model.layers.L.mlp / language_model.mtp.layers.0.mlp: the VLM tree's module paths."""

    def __init__(self, args):
        super().__init__()
        self.language_model = _LM(args)


def _args():
    return QL.Qwen4ExpTextArgs(hidden_size=D, num_experts=E, num_experts_per_tok=TOPK, moe_intermediate_size=I,
                               shared_expert_intermediate_size=SHARED)


def _config(plan=PLAN, drop=None, jangtq=False):
    quant = {"group_size": 64, "bits": 4, "mode": "affine"}
    outl = {}
    for L, (kgu, kdn) in plan.items():
        m = f"model.layers.{L}.mlp.switch_mlp."
        for p, K, n in (("gate_proj", kgu, N_GU), ("up_proj", kgu, N_GU), ("down_proj", kdn, N_DN)):
            if (L, p) == drop:
                continue
            quant[m + p] = {"mode": "jangt", "kbits": K}
            outl[m + p] = n
    cfg = {"jangt": {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                     "outlier_columns": outl}, "quantization": quant}
    if jangtq:   # JANGHT bundles assembled from a JANGH base keep its declaration (codebooks of JANGH groups)
        cfg["jangtq"] = {"version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "codebook_family": "odd-cubic",
                         "rotation": "hadamard32"}
    return cfg


_BUILT = {}


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _build():
    """Model with JANGT installed and loaded from synthetic encoder stage tensors + the dense reference weights."""
    if "m" in _BUILT:
        return _BUILT["m"]
    mx.random.seed(0)
    rng = np.random.default_rng(2026)
    root = _Root(_args())
    # router / shared expert / MTP weights: random dense values (the checkpoint's non-JANGT tensors)
    dense = {k: mx.array((rng.standard_normal(v.shape) * (0.02 if "gate." not in k else 0.05)).astype(np.float32))
             for k, v in tree_flatten(root.parameters())}
    mtp_before = {k: v for k, v in dense.items() if ".mtp." in k}
    jt_n, jh_n = QLD._install_routed_expert_formats(root, _config(jangtq=True))
    assert (jt_n, jh_n) == (len(PLAN), 0)
    units, refs, weights = {}, {}, {}
    for L, (kgu, kdn) in PLAN.items():
        P = S.layer_unit(rng, D, I, E, kgu, kdn, N_GU, N_DN, VL)
        units[L] = P
        refs[L] = {p: S.dense_reference(P[p], REF) for p in ("gate", "up", "down")}
        for p in ("gate", "up", "down"):
            # on-disk names of a JANGHT bundle / the encoder stage files
            weights.update(S.stage_tensors(f"model.layers.{L}.mlp.switch_mlp.{p}_proj", P[p]))
    # non-routed tensors under their checkpoint (already sanitized) names
    for k, v in dense.items():
        if ".switch_mlp." in k and ".mtp." not in k:
            continue                                                  # backbone routed experts are JANGT now
        weights[k.replace("language_model.model.", "language_model.", 1)] = v
    weights = QLD._normalize_runtime_weight_names(weights)
    assert SW.normalize_jangt_payload(root, weights) == 3 * len(PLAN)
    assert not any(k.endswith((".jt_off", ".jt_kbits")) for k in weights)
    summary = QLD._normalize_jang_runtime_compute_dtypes(weights, root, mx.bfloat16)
    assert summary["preserved"] == 10 * len(PLAN)          # jt_scale/jt_su/jt_wout x3 projections + down jt_cs
    for k, v in weights.items():                                      # format tensors keep their dtypes
        if SW.is_jangt_format_tensor(k):
            leaf = k.rsplit(".", 1)[-1]
            want = {"jt_packed": mx.uint32, "jt_ocol": mx.int32, "jt_wout": mx.float16}.get(leaf, mx.float32)
            assert v.dtype == want, (k, v.dtype)
    QLD._quantize_model(root, _config(), weights)                    # JTSwitchLinear must be left alone
    root.load_weights(list(weights.items()), strict=True)
    mx.eval(root.parameters())
    _BUILT["m"] = (root, units, refs, mtp_before)
    return _BUILT["m"]


def _block_reference(block, refs, x):
    """Dense float64 qwen4_exp SparseMoeBlock output; routing taken from the block's own (unchanged) router."""
    gates = mx.softmax(block.gate(mx.array(x)), axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-TOPK, axis=-1)[..., -TOPK:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    idx = np.array(inds).reshape(-1, TOPK); w = np.array(scores).astype(np.float64).reshape(-1, TOPK)
    x2 = x.reshape(-1, D).astype(np.float64)
    _, routed = S.moe_reference(refs["gate"], refs["up"], refs["down"], x2, idx, w)
    se = block.shared_expert
    wg, wu, wd = (_np(l.weight) for l in (se.gate_proj, se.up_proj, se.down_proj))
    shared = np.einsum("dh,th->td", wd, S.silu(np.einsum("hd,td->th", wg, x2)) * np.einsum("hd,td->th", wu, x2))
    sg = 1.0 / (1.0 + np.exp(-np.einsum("d,td->t", _np(block.shared_expert_gate.weight)[0], x2)))
    return routed + sg[:, None] * shared


def test_install_replaces_backbone_routed_experts_only():
    root, units, refs, mtp_before = _build()
    for L, (kgu, kdn) in PLAN.items():
        sw = root.language_model.model.layers[L].mlp.switch_mlp
        assert isinstance(sw, SW.JTMixedSwitchGLU) and sw.is_jangt
        assert (sw.gate_proj.kbits, sw.up_proj.kbits, sw.down_proj.kbits) == (kgu, kgu, kdn)
        assert sw.gate_proj.jt_packed.shape == (E, I, VL.words_per_row(D, 12, kgu))
        assert sw.down_proj.jt_packed.shape == (E, D, VL.words_per_row(I, 12, kdn))
        np.testing.assert_array_equal(np.array(sw.down_proj.jt_packed), units[L]["down"]["packed"])
        blk = root.language_model.model.layers[L].mlp
        assert isinstance(blk.shared_expert, QL.SharedExpertMLP) and isinstance(blk.gate, nn.Linear)
    mtp = root.language_model.mtp.layers[0].mlp.switch_mlp
    assert isinstance(mtp, SwitchGLU) and not getattr(mtp, "is_jangt", False)
    after = dict(tree_flatten(root.parameters()))
    for k, v in mtp_before.items():          # MTP experts loaded unchanged (affine path; compute dtype bf16 like the rest)
        assert after[k].dtype == mx.bfloat16
        np.testing.assert_array_equal(_np(after[k]), _np(v.astype(mx.bfloat16)))


@pytest.mark.parametrize("L", sorted(PLAN))
@pytest.mark.parametrize("T,route", [(3, "decode"), (12, "prefill")])
def test_block_matches_dense_reference(L, T, route):
    root, units, refs, _ = _build()
    block = root.language_model.model.layers[L].mlp
    rng = np.random.default_rng(10 * L + T)
    x = (rng.standard_normal((1, T, D)) * 0.5).astype(np.float32)
    assert (T * TOPK >= SW.SORT_THRESHOLD) == (route == "prefill")
    calls = []
    orig = block.switch_mlp._prefill
    block.switch_mlp._prefill = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        y = block(mx.array(x))
        mx.eval(y)
    finally:
        del block.switch_mlp._prefill
    took_prefill = bool(calls)
    assert took_prefill == (route == "prefill" and JK.nax_available())
    ref = _block_reference(block, refs[L], x)
    err = S.rel_l2(np.array(y.astype(mx.float32)).reshape(T, D), ref)
    # decode kernels: fp32 x, half2 inner products (~7e-4 measured); NAX prefill: bf16 activations and weight tiles
    bound = 2e-2 if took_prefill else 3e-3
    print(f"layer {L} K={PLAN[L]} T={T} route={'nax-prefill' if took_prefill else 'decode'}: rel-L2 {err:.2e}")
    assert err < bound


def test_jangt_owns_routed_stacks_even_with_a_jangh_declaration():
    root = _Root(_args())
    assert QLD._install_routed_expert_formats(root, _config(jangtq=True)) == (len(PLAN), 0)


def test_incomplete_contract_fails_closed():
    root = _Root(_args())
    with pytest.raises(ValueError, match="incomplete routed contract"):
        QLD._install_routed_expert_formats(root, _config(drop=(1, "down_proj")))


def test_stage_width_disagreeing_with_contract_fails_closed():
    root = _Root(_args())
    QLD._install_routed_expert_formats(root, _config())
    rng = np.random.default_rng(1)
    # the stage file holds a K=2 gate for layer 0 while the contract says 1.5 (words_per_row agrees by accident
    # nowhere, but the width check must not depend on that)
    P = S.random_projection(rng, E, I, D, 2, N_GU, True, False, VL)
    w = QLD._normalize_runtime_weight_names(S.stage_tensors("model.layers.0.mlp.switch_mlp.gate_proj", P))
    with pytest.raises(ValueError, match="do not match the contract"):
        SW.normalize_jangt_payload(root, w)


def test_no_jangt_block_keeps_the_previous_routes():
    root = _Root(_args())
    assert QLD._install_routed_expert_formats(root, {"quantization": {"bits": 4}}) == (0, 0)
    assert all(isinstance(l.mlp.switch_mlp, SwitchGLU) for l in root.language_model.model.layers)

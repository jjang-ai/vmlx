# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 / M3.1 runtime defaults (2026-10-29): fused attention (``VMLX_M3_FUSED_ATTN``), fast JANGT decode kernels
(``VMLX_JANGT_FAST``) and E1 exact multi-row verify (``VMLX_M3_EXACT_FLAT``) are ON with no env, each turned off by its
own ``=0``; fast JANGT is scoped to M3's routed stacks (marked at load by ``minimax_m3.Model``), so a JANGT unit of any
other family keeps the v3 kernels unless ``VMLX_JANGT_FAST=1``.

The M3 model here is built through the real load class (``Model`` with a ``jangt`` block: ``install_jangt`` then the
per-model fast default), with M3.1's formats where the three paths apply: q / k / v / o affine-4 g64 and MXFP8 index
projections (fused attention + its E1 K1 / K5), affine-4 shared experts / dense MLP / lm_head (E1 in the verify),
JANGT routed experts at whole-512 widths (fast kernels). The paths taken are asserted with the modules' own counters
(``fused_attn.STATS`` / ``E1_STATS``, ``fast_decode.STATS``, the DSpark target's ``verify_paths``), and every arm keeps
verify rows == decode steps bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

import tests.test_jangt_grouped_decode as TG  # noqa: E402
import tests.test_minimax_m3_fused_moe as TM  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402
from vmlx_engine.jangt import fast_decode as FD  # noqa: E402
from vmlx_engine.jangt import switch as SW  # noqa: E402
from vmlx_engine.metal import row_exact_e1 as E1  # noqa: E402
from vmlx_engine.models.minimax_m3 import fused_attn as F  # noqa: E402
from vmlx_engine.models.minimax_m3 import minimax_m3 as M  # noqa: E402

D, I, E = 1024, 512, TM.E
WIDTHS = {1: (1.5, 2.5, 1, 3), 2: (2, 1.5, 2, 4)}       # MoE layer -> (gate/up K, down K, gate/up outliers, down outliers)
MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}
A4 = {"group_size": 64, "bits": 4}
FAST, ATTN, FLAT = FD.ENV, F.ENV, F.EXACT_FLAT_ENV
LEVERS = (FAST, ATTN, FLAT, "VMLX_M3_FUSED_MOE", "VMLX_M3_E1_BITS", "VMLX_M3_E1_SHAPE", "VMLX_E1_QMV_SHAPE",
          "VMLX_ROW_EXACT_VERIFY_QMV", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH", "VMLX_DSPARK_SAMPLED",
          "VMLX_DSPARK_PIPELINE", "VMLX_DSPARK_AR_FALLBACK", "VMLX_MINIMAX_MSA", "VMLX_MINIMAX_MSA_SELECT")


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """Every arm starts from NO lever env (the new defaults); a test sets exactly the lever it varies."""
    for name in LEVERS:
        monkeypatch.delenv(name, raising=False)


def _config() -> dict:
    tc = {"model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": 3, "intermediate_size": I,
          "dense_intermediate_size": 1024, "shared_intermediate_size": I, "num_attention_heads": 8,
          "num_key_value_heads": 2, "head_dim": 64, "rotary_dim": 32, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
          "vocab_size": 512, "num_local_experts": E, "num_experts_per_tok": 4, "routed_scaling_factor": 2.0,
          "norm_topk_prob": True, "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "moe_layer_freq": [0, 1, 1],
          "sparse_attention_config": {"sparse_num_index_heads": 2, "sparse_index_dim": 64, "sparse_block_size": 32,
                                      "sparse_topk_blocks": 3, "sparse_local_block": 1,
                                      "sparse_attention_freq": [1, 1, 1]}}
    quant, outl = {"group_size": 64, "bits": 4}, {}
    for L, (kg, kd, nog, nod) in WIDTHS.items():
        key = f"model.layers.{L}.mlp.switch_mlp"
        quant.update({f"{key}.gate_proj": {"mode": "jangt", "kbits": kg}, f"{key}.up_proj": {"mode": "jangt", "kbits": kg},
                      f"{key}.down_proj": {"mode": "jangt", "kbits": kd}})
        outl.update({f"{key}.gate_proj": nog, f"{key}.up_proj": nog, f"{key}.down_proj": nod})
    return {"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "text_config": tc, "quantization": quant,
            "jangt": {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                      "outlier_columns": outl}}


def _plan(path, module):
    if path.endswith("mlp.gate") or "switch_mlp" in path:
        return False                                   # fp32 router; routed experts are the installed JANGT units
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    return MX8 if "index_" in path else A4


def _build():
    from mlx.utils import tree_flatten

    model = M.Model(M.ModelArgs.from_dict(_config()))   # the load class: install_jangt + the M3 fast default
    rng = np.random.default_rng(0)
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if "switch_mlp" in k:
            continue
        if k.endswith("norm.weight"):
            a, dt = rng.standard_normal(v.shape) * 0.1, mx.bfloat16
        elif k.endswith(("e_score_correction_bias", ".mlp.gate.weight")):
            a, dt = rng.standard_normal(v.shape) * 0.05, mx.float32
        elif "embed_tokens" in k:
            a, dt = rng.standard_normal(v.shape), mx.bfloat16
        else:
            a, dt = rng.standard_normal(v.shape) * 1.5 / np.sqrt(v.shape[-1]), mx.bfloat16
        weights.append((k, mx.array(a.astype(np.float32)).astype(dt)))
    model.load_weights(weights, strict=False)
    for L, (kg, kd, nog, nod) in WIDTHS.items():
        # random trellis tensors bound INTO the installed units (the load-time marking stays on the same objects)
        model.model.layers[L].mlp.switch_mlp.update(TM._jangt(D, I, kg, kd, nog, nod, rng).parameters())
    nn.quantize(model, class_predicate=_plan)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model():
    return _build()


@pytest.fixture(scope="module")
def target(model):
    t = MiniMaxM3DSparkTarget(model)
    assert [F.unsupported_reason(layer) for layer in t.nm.layers] == [None, None, None]
    for L in WIDTHS:
        sw = t.nm.layers[L].mlp.switch_mlp
        assert FD.supported(sw._unit(), D, I)
    return t


PROMPT = np.random.default_rng(7).integers(0, 511, 141).tolist()
ROWS = np.random.default_rng(8).integers(0, 511, 8).tolist()


def _eq(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def _counters(t):
    return {"attn": F.STATS["calls"], "e1_qkv": F.E1_STATS["qkv"], "e1_oproj": F.E1_STATS["oproj"],
            "fast": FD.STATS["calls"], **{f"v_{k}": v for k, v in t.verify_paths.items()}}


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def _verify_and_steps(t):
    """(verify logits of ROWS after PROMPT, the counters the verify moved, the decode-step logits, the counters ONE
    decode step moved). Counters are read around the verify / the first step only (the prefill is excluded)."""
    rows = mx.array([ROWS])
    cache = t.make_cache()
    t.prefill(mx.array([PROMPT]), cache, (), 32)
    before = _counters(t)
    logits, _, _ = t.verify(rows, cache, ())
    mx.eval(logits)
    dv = _delta(_counters(t), before)
    ref = t.make_cache()
    t.prefill(mx.array([PROMPT]), ref, (), 32)
    before = _counters(t)
    steps = [t.decode(rows[0, :1], ref)]
    mx.eval(steps)
    ds = _delta(_counters(t), before)
    steps += [t.decode(rows[0, r:r + 1], ref) for r in range(1, len(ROWS))]
    mx.eval(steps)
    return logits, dv, steps, ds


# ------------------------------------------------------------------------------------------------ the M3 defaults
def test_m3_load_marks_its_jangt_units_fast_by_default(model, monkeypatch):
    units = [mod for _, mod in model.named_modules() if isinstance(mod, SW.JTMixedSwitchGLU)]
    assert len(units) == len(WIDTHS) == model.jangt_modules
    assert all(getattr(u, FD.DEFAULT_ATTR, False) is True for u in units)
    assert all(FD.enabled(u) for u in units)                         # no env: on
    monkeypatch.setenv(FAST, "0")
    assert not any(FD.enabled(u) for u in units)
    monkeypatch.setenv(FAST, "1")
    assert all(FD.enabled(u) for u in units)


def test_lever_functions_default_on_and_off_with_zero(monkeypatch):
    assert F.enabled() and F.exact_flat() and E1.enabled()
    for name, fn in ((ATTN, F.enabled), (FLAT, F.exact_flat), (FLAT, E1.enabled)):
        for off in ("0", "false", "OFF", " no "):
            monkeypatch.setenv(name, off)
            assert not fn(), (name, off)
        for on in ("1", "true", "on", "yes", ""):
            monkeypatch.setenv(name, on)
            assert fn(), (name, on)
        monkeypatch.delenv(name)


def test_m3_no_env_takes_fused_attention_fast_jangt_and_e1(target):
    """No env: a DSpark verify of 8 rows runs every layer's attention fused (3 layers) with the E1 K1 / K5 builds, the
    routed experts of both MoE layers on the fast JANGT kernels, the other affine verify projections on E1; a decode
    step runs fused attention + fast JANGT (and no E1: one row). Verify rows == decode steps bit for bit."""
    logits, dv, steps, ds = _verify_and_steps(target)
    assert dv["attn"] == 3 and dv["v_attention_fused"] == 3, dv
    assert dv["e1_qkv"] == 3 and dv["e1_oproj"] == 3, dv
    assert dv["fast"] == 2 and dv["v_routed_batched"] == 2 and dv["v_routed_rows"] == 0, dv
    assert dv["v_linear_e1"] > 0, dv
    assert ds["attn"] == 3 and ds["fast"] == 2, ds
    assert ds["e1_qkv"] == 0 and ds["e1_oproj"] == 0, ds
    assert all(_eq(logits[r], steps[r]) for r in range(len(ROWS)))


def test_m3_explicit_on_equals_the_default(target, monkeypatch):
    """``=1`` for all three gives the same bits as no env (the default IS the lever on)."""
    default, _, _, _ = _verify_and_steps(target)
    for name in (FAST, ATTN, FLAT):
        monkeypatch.setenv(name, "1")
    explicit, _, _, _ = _verify_and_steps(target)
    assert _eq(default, explicit)


@pytest.mark.parametrize("lever", [ATTN, FAST, FLAT])
def test_m3_each_lever_zero_turns_only_its_path_off(target, lever, monkeypatch):
    monkeypatch.setenv(lever, "0")
    logits, dv, steps, ds = _verify_and_steps(target)
    if lever == ATTN:
        assert dv["attn"] == 0 and ds["attn"] == 0 and dv.get("v_attention_fused", 0) == 0, dv
        assert dv["e1_qkv"] == 0 and dv["e1_oproj"] == 0, dv          # K1 / K5 E1 builds live in the fused path
        assert dv["fast"] == 2 and ds["fast"] == 2 and dv["v_linear_e1"] > 0, dv
    elif lever == FAST:
        assert dv["fast"] == 0 and ds["fast"] == 0, (dv, ds)
        assert dv["attn"] == 3 and dv["e1_qkv"] == 3 and dv["v_linear_e1"] > 0, dv
    else:
        assert dv["e1_qkv"] == 0 and dv["e1_oproj"] == 0 and dv.get("v_linear_e1", 0) == 0, dv
        assert dv["attn"] == 3 and dv["fast"] == 2, dv
    assert all(_eq(logits[r], steps[r]) for r in range(len(ROWS))), lever


def test_m3_all_three_zero_is_the_old_path(target, monkeypatch):
    for name in (FAST, ATTN, FLAT):
        monkeypatch.setenv(name, "0")
    logits, dv, steps, ds = _verify_and_steps(target)
    for d in (dv, ds):
        assert d["attn"] == 0 and d["fast"] == 0 and d["e1_qkv"] == 0 and d["e1_oproj"] == 0, d
    assert dv.get("v_linear_e1", 0) == 0 and dv.get("v_attention_fused", 0) == 0, dv
    assert all(_eq(logits[r], steps[r]) for r in range(len(ROWS)))
    for name in (FAST, ATTN, FLAT):
        monkeypatch.delenv(name)
    new, _, _, _ = _verify_and_steps(target)
    assert not _eq(new, logits)          # control: the defaults really changed the arithmetic (not bit-identical)


# ------------------------------------------------------------------------------------------------ other JANGT families
def test_non_m3_jangt_unit_keeps_v3_without_env(monkeypatch):
    """A JANGT unit no M3 load marked (every other family): no env runs the v3 kernels -- bit-identical to ``=0`` and
    no fast launch; ``=1`` still forces the fast kernels (control: it moves the counter and the bits)."""
    m = TG._module(1024, 512, 1.5, 2.5, 2, 7.0, "oai", seed=11)
    assert FD.supported(m._unit(), 1024, 512) and not hasattr(m, FD.DEFAULT_ATTR)
    idx, w, _ = TG._routing(8, 4, seed=11, mode="pool")
    x = TG._x(8, 1024, 11)[None]
    calls = FD.STATS["calls"]
    default = m.routed(x, idx[None], w[None])
    one_row = m.routed(x[:, :1], idx[None, :1], w[None, :1])
    mx.eval(default, one_row)
    assert FD.STATS["calls"] == calls
    monkeypatch.setenv(FAST, "0")
    assert _eq(default, m.routed(x, idx[None], w[None]))
    assert _eq(one_row, m.routed(x[:, :1], idx[None, :1], w[None, :1]))
    monkeypatch.setenv(FAST, "1")
    forced = m.routed(x, idx[None], w[None])
    mx.eval(forced)
    assert FD.STATS["calls"] == calls + 1 and not _eq(forced, default)


def test_non_m3_family_load_does_not_mark_units():
    """GLM-5.3's text Model installs JANGT stacks through the same install_jangt and stays unmarked (v3 by default)."""
    import tests.test_glm5_text_only_jangt as TGL
    from vmlx_engine.models.glm5_next.glm5_next import Model, ModelArgs

    m = Model(ModelArgs.from_dict({"model_type": "glm5_next", "text_config": dict(TGL.TEXT), "jangt": TGL.JANGT,
                                   "quantization": TGL._quant()}))
    units = [mod for _, mod in m.named_modules() if isinstance(mod, SW.JTMixedSwitchGLU)]
    assert len(units) == 3
    assert not any(getattr(u, FD.DEFAULT_ATTR, False) for u in units)
    assert not any(FD.enabled(u) for u in units)


def test_only_minimax_m3_enables_the_fast_default():
    """The per-model default is set by MiniMax-M3's load and nothing else in the engine."""
    root = Path(SW.__file__).resolve().parents[1]
    callers = sorted(str(p.relative_to(root)) for p in root.rglob("*.py")
                     if "enable_fast_decode_default(" in p.read_text(errors="replace"))
    assert callers == ["jangt/switch.py", "models/minimax_m3/minimax_m3.py"], callers

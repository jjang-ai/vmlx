# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 DSpark drafter, pre-quantized: write_quantized_minimax_drafter -> load_drafter.

A tiny drafter is written in the release layout (``language_model.model.dspark.layers.N.decoder_layer.*`` names,
decoder linears MXFP8 as e4m3 bytes + ``weight_scale_inv`` e8m0 scales, fc / hidden_norm / final_norm / Markov /
confidence heads BF16). Checked:
  - the release directory still loads as stored (same bytes, same modules, same output as the previous loader);
  - a quantized copy (per-module plan: affine 4/6/8-bit, MXFP8 modules kept as stored, BF16 heads) loads with
    exactly its config.json["quantization"] entries (keyed by the drafter's live module paths) and is bit-identical
    to nn.quantize of the same model in memory: block forward and the Markov draft (candidate and full-vocabulary
    paths);
  - quantized Markov / confidence heads draft like their dequantized float weights (control: reading the packed words
    as weights, as the head code did before, does not);
  - the writer never touches the source.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("jang_tools.format.aligned_safetensors")

from mlx.utils import tree_flatten  # noqa: E402

from vmlx_engine.dspark import drafter as DR  # noqa: E402

HID, HEADS, KV, HD, ROT, FFN, TAPS, VOCAB, RANK = 64, 4, 2, 16, 8, 128, 2, 128, 64
WINDOW, BLOCK, CTX, LAYERS = 6, 4, 9, 2
PRE = "language_model.model.dspark."
MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}
A = lambda b: {"group_size": 64, "bits": b}  # noqa: E731
PLANS = {
    "default": None,                                                     # MINIMAX_DRAFTER_PLAN
    "markov8": {"layers.*": A(4), "fc": A(8), "markov_head.*": A(8), "confidence_head.*": None},
    "mixed": {"layers.1.self_attn.*": MX8, "layers.0.mlp.down_proj": A(6), "layers.*": A(4), "fc": A(8),
              "markov_head.markov_w2": A(8), "markov_head.*": None, "confidence_head.*": None},
}


def _raw_config():
    return {"architectures": ["DSparkMiniMaxDraftModel"], "enable_confidence_head": True,
            "confidence_head_with_markov": True, "text_config": {
                "hidden_size": HID, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
                "num_key_value_heads": KV, "head_dim": HD, "rotary_dim": ROT, "rms_norm_eps": 1e-6,
                "vocab_size": VOCAB, "dspark_ffn_hidden_size": FFN, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
                "sliding_window": WINDOW, "dspark_block_size": BLOCK, "rope_theta": 10_000.0,
                "dspark_noise_token_id": VOCAB - 1, "dspark_target_layer_ids": [1, 3], "dspark_markov_rank": RANK,
                "layer_types": ["sliding_attention"] * LAYERS, "dense_intermediate_size": FFN}}


def _mxfp8(rng, out_d, in_d):
    """e4m3 bytes (exponent field 5..8: 0.25-3.75) and e8m0 scales 2^-6..2^-4, like the release's linears."""
    sign = rng.integers(0, 2, (out_d, in_d)) << 7
    w = (sign | (rng.integers(5, 9, (out_d, in_d)) << 3) | rng.integers(0, 8, (out_d, in_d))).astype(np.uint8)
    return mx.array(w), mx.array(rng.integers(121, 124, (out_d, in_d // 32)).astype(np.uint8))


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    rng = np.random.default_rng(0)
    bf = lambda *s, k=0.3: mx.array((rng.standard_normal(s) * k).astype(np.float32)).astype(mx.bfloat16)  # noqa: E731
    t = {PRE + "fc.weight": bf(HID, TAPS * HID, k=0.1), PRE + "hidden_norm.weight": bf(HID, k=0.1),
         PRE + "final_norm.weight": bf(HID, k=0.1), PRE + "markov_head.markov_w1.weight": bf(VOCAB, RANK),
         PRE + "markov_head.markov_w2.weight": bf(VOCAB, RANK), PRE + "confidence_head.proj.weight": bf(1, HID + RANK),
         PRE + "confidence_head.proj.bias": bf(1)}
    for L in range(LAYERS):
        p = f"{PRE}layers.{L}.decoder_layer."
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            t[p + n + ".weight"] = bf(HD if "_norm" in n and "self_attn" in n else HID, k=0.1)
        for n, o, i in (("self_attn.q_proj", HEADS * HD, HID), ("self_attn.k_proj", KV * HD, HID),
                        ("self_attn.v_proj", KV * HD, HID), ("self_attn.o_proj", HID, HEADS * HD),
                        ("mlp.gate_proj", FFN, HID), ("mlp.up_proj", FFN, HID), ("mlp.down_proj", HID, FFN)):
            t[p + n + ".weight"], t[p + n + ".weight_scale_inv"] = _mxfp8(rng, o, i)
    d = tmp_path_factory.mktemp("dspark_src")
    mx.save_safetensors(str(d / "model.safetensors"), t, metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps(_raw_config(), indent=1))
    (d / "README.md").write_text("tiny drafter\n")
    return d


def _digest(d):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.iterdir()) if f.is_file()}


def _inputs():
    rng = np.random.default_rng(1)
    feats = mx.array(rng.standard_normal((1, CTX, TAPS * HID)).astype(np.float32)).astype(mx.bfloat16)
    block = mx.array(rng.standard_normal((1, BLOCK, HID)).astype(np.float32)).astype(mx.bfloat16)
    head = nn.Linear(HID, VOCAB, bias=False)
    head.weight = mx.array(rng.standard_normal((VOCAB, HID)).astype(np.float32) * 0.3).astype(mx.bfloat16)
    return feats, block, head


def _outputs(d):
    """Block forward + the Markov draft on the candidate path (K=16) and the full-vocabulary path (K=0)."""
    feats, block, head = _inputs()
    d.reset()
    d.add_context(feats)
    h = d(block)
    anchor = mx.array([5])
    out = {"h": h}
    for k in (16, 0):
        tok, conf = d.draft(h, anchor, head, BLOCK, candidates=k)
        out[f"tok{k}"], out[f"conf{k}"] = tok, conf
    mx.eval(out)
    return out


def _same(a, b):
    return all(mx.array_equal(a[k], b[k]).item() for k in a)


def _spec(plan, path):
    for pattern, spec in plan.items():
        if fnmatch.fnmatchcase(path, pattern):
            return spec
    return None


def _in_memory(src, plan):
    """The same drafter quantized in memory: the release loaded as stored, MXFP8 dequantized to BF16, nn.quantize with
    the plan's affine entries; modules the plan keeps MXFP8 are the stored modules themselves."""
    plan = DR.MINIMAX_DRAFTER_PLAN if plan is None else plan
    stored = DR.load_drafter(src)
    flt = DR.DSparkDrafter(DR.minimax_config(_raw_config()))
    mods = dict(stored.named_modules())
    weights = []
    for k, v in tree_flatten(stored.parameters()):
        base, _, leaf = k.rpartition(".")
        m = mods.get(base)
        if isinstance(m, nn.QuantizedLinear):
            if leaf == "weight":
                weights.append((k, mx.dequantize(m.weight, m.scales, None, group_size=32, bits=8, mode="mxfp8",
                                                 dtype=mx.bfloat16)))
            continue
        weights.append((k, v))
    flt.load_weights(weights, strict=True)
    keep = [p for p, m in mods.items() if isinstance(m, nn.QuantizedLinear)
            and (_spec(plan, p) is None or _spec(plan, p).get("mode") == "mxfp8")]
    nn.quantize(flt, class_predicate=lambda p, m: (dict(_spec(plan, p)) if isinstance(m, (nn.Linear, nn.Embedding))
                                                   and p not in keep and _spec(plan, p)
                                                   and _spec(plan, p).get("mode", "affine") == "affine" else False))
    for p in keep:
        parent, _, leaf = p.rpartition(".")
        setattr(DR._module_at(flt, parent), leaf, mods[p])
    mx.eval(flt.parameters())
    return flt


def _quant_map(model):
    out = {}
    for p, m in model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            out[p] = ({"group_size": m.group_size, "bits": m.bits} if m.mode == "affine"
                      else {"mode": m.mode, "group_size": m.group_size, "bits": m.bits})
    return out


# ------------------------------------------------------------------------------------------------ release layout
def test_release_directory_loads_as_stored(src):
    """No quantization block: MXFP8 linears keep the stored bytes, everything else BF16 -- the previous loader."""
    from vmlx_engine.models.minimax_m3.source_format import convert_mxfp8_linears

    d = DR.load_drafter(src)
    assert set(_quant_map(d)) == {f"layers.{L}.{n}" for L in range(LAYERS) for n in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    assert all(e == MX8 for e in _quant_map(d).values())
    raw = mx.load(str(src / "model.safetensors"))
    q = d.layers[1].self_attn.q_proj
    assert mx.array_equal(q.weight, mx.view(raw[PRE + "layers.1.decoder_layer.self_attn.q_proj.weight"], mx.uint32)).item()
    assert mx.array_equal(q.scales, raw[PRE + "layers.1.decoder_layer.self_attn.q_proj.weight_scale_inv"]).item()
    assert isinstance(d.fc, nn.Linear) and d.fc.weight.dtype == mx.bfloat16
    # the loader as it was before pre-quantized directories existed
    old = DR.DSparkDrafter(DR.minimax_config(_raw_config()))
    renamed = {}
    for k, v in raw.items():
        k = k.removeprefix(PRE).replace(".decoder_layer.", ".")
        renamed["norm." + k[len("final_norm."):] if k.startswith("final_norm.") else k] = v
    convert_mxfp8_linears(old, renamed)
    old.load_weights(list(renamed.items()), strict=True)
    assert _same(_outputs(d), _outputs(old))


# ------------------------------------------------------------------------------------------------ quantized copies
@pytest.mark.parametrize("name", list(PLANS))
def test_quantized_copy_equals_in_memory_quantization(src, tmp_path, name):
    before = _digest(src)
    dst = DR.write_quantized_minimax_drafter(src, tmp_path / "q", PLANS[name])
    assert _digest(src) == before                                             # never touches the source
    assert sorted(f.name for f in dst.iterdir()) == ["README.md", "config.json", "model.safetensors"]
    cfg = json.loads((dst / "config.json").read_text())
    quant = cfg["quantization"]
    entries = {k: v for k, v in quant.items() if isinstance(v, dict)}
    loaded = DR.load_drafter(dst)
    assert _quant_map(loaded) == entries                                      # every module carries its entry
    mem = _in_memory(src, PLANS[name])
    assert _quant_map(mem) == entries
    lp, mp = dict(tree_flatten(loaded.parameters())), dict(tree_flatten(mem.parameters()))
    assert set(lp) == set(mp) and all(lp[k].dtype == mp[k].dtype and mx.array_equal(lp[k], mp[k]).item() for k in lp)
    assert _same(_outputs(loaded), _outputs(mem))
    # the copy really is a different model from the release (4-bit layers), so the comparison above is not vacuous
    assert not mx.array_equal(_outputs(loaded)["h"], _outputs(DR.load_drafter(src))["h"]).item()
    # the key form: live module paths, no release names
    assert all(not k.startswith("language_model.") and ".decoder_layer." not in k for k in entries)
    assert "layers.0.self_attn.q_proj" in entries and ("fc" in entries)
    if name == "mixed":
        assert entries["layers.1.self_attn.q_proj"] == MX8 and entries["layers.0.mlp.down_proj"] == A(6)
        stored_q = DR.load_drafter(src).layers[1].self_attn.q_proj
        assert mx.array_equal(loaded.layers[1].self_attn.q_proj.weight, stored_q.weight).item()   # MXFP8 bytes kept
        assert "markov_head.markov_w2" in entries and "markov_head.markov_w1" not in entries


def test_entry_missing_falls_back_to_the_top_level_default(src, tmp_path):
    """mlx-lm semantics: a module with .scales and no entry takes the block's group_size / bits."""
    dst = DR.write_quantized_minimax_drafter(src, tmp_path / "q", PLANS["markov8"])
    cfg = json.loads((dst / "config.json").read_text())
    assert (cfg["quantization"]["group_size"], cfg["quantization"]["bits"]) == (64, 4)
    full = _outputs(DR.load_drafter(dst))
    del cfg["quantization"]["layers.0.self_attn.q_proj"]
    (dst / "config.json").write_text(json.dumps(cfg))
    d = DR.load_drafter(dst)
    assert _quant_map(d)["layers.0.self_attn.q_proj"] == A(4)
    assert _same(_outputs(d), full)


def test_mxfp8_pairs_in_release_form_inside_a_quantized_directory(src, tmp_path):
    """A pre-quantized directory may keep some MXFP8 linears in the release form (weight + weight_scale_inv, no entry):
    they load as MXFP8 next to the affine modules."""
    dst = DR.write_quantized_minimax_drafter(src, tmp_path / "q", PLANS["mixed"])
    ref = _outputs(DR.load_drafter(dst))
    w = dict(mx.load(str(dst / "model.safetensors")))
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        base = f"layers.1.self_attn.{n}"
        w[base + ".weight"] = mx.view(w[base + ".weight"], mx.uint8)
        w[base + ".weight_scale_inv"] = w.pop(base + ".scales")
    (dst / "model.safetensors").unlink()
    mx.save_safetensors(str(dst / "model.safetensors"), w)
    cfg = json.loads((dst / "config.json").read_text())
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        del cfg["quantization"][f"layers.1.self_attn.{n}"]
    (dst / "config.json").write_text(json.dumps(cfg))
    d = DR.load_drafter(dst)
    assert _quant_map(d)["layers.1.self_attn.k_proj"] == MX8
    assert _same(_outputs(d), ref)


# ------------------------------------------------------------------------------------------------ quantized heads
def test_quantized_markov_and_confidence_heads_draft_like_their_float_values(src, tmp_path, monkeypatch):
    plan = {"layers.*": A(4), "fc": A(8), "markov_head.*": A(8), "confidence_head.*": A(8)}
    q = DR.load_drafter(DR.write_quantized_minimax_drafter(src, tmp_path / "q", plan))
    f = DR.load_drafter(DR.write_quantized_minimax_drafter(src, tmp_path / "f", plan))
    for name in ("markov_head.markov_w1", "markov_head.markov_w2", "confidence_head.proj"):
        m = dict(q.named_modules())[name]
        parent, _, leaf = name.rpartition(".")
        if isinstance(m, nn.QuantizedEmbedding):
            fl = nn.Embedding(VOCAB, RANK)
        else:
            fl = nn.Linear(m.weight.shape[1] * 32 // m.bits, m.weight.shape[0], bias="bias" in m)
            if "bias" in m:
                fl.bias = m.bias
        fl.weight = DR._float_weight(m)
        setattr(DR._module_at(f, parent), leaf, fl)
    a, b = _outputs(q), _outputs(f)
    assert mx.array_equal(a["h"], b["h"]).item()
    for k in (16, 0):
        assert mx.array_equal(a[f"tok{k}"], b[f"tok{k}"]).item(), k
        np.testing.assert_allclose(np.array(a[f"conf{k}"]), np.array(b[f"conf{k}"]), rtol=1e-3, atol=1e-4)
    # control: the head code before this change read .weight directly (packed words for a quantized head)
    monkeypatch.setattr(DR, "_rows", lambda m, idx: m.weight if idx is None else m.weight[idx])
    try:
        c = _outputs(q)
        bites = not (mx.array_equal(c["tok16"], b["tok16"]).item() and np.allclose(np.array(c["conf16"]),
                                                                                    np.array(b["conf16"]), atol=1e-4))
    except (ValueError, RuntimeError):
        bites = True
    assert bites


def test_writer_refuses_a_non_empty_destination_and_a_foreign_drafter(src, tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        DR.write_quantized_minimax_drafter(src, tmp_path / "busy")
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"architectures": ["Qwen3DSparkModel"]}))
    with pytest.raises(ValueError):
        DR.write_quantized_minimax_drafter(other, tmp_path / "out")

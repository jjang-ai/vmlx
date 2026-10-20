# SPDX-License-Identifier: Apache-2.0
"""Split per-expert routed stacks in the JANG v2 VLM loader.

Ornith-1.5-122B stores its native-MTP head's routed experts one tensor per expert
(mtp.layers.0.mlp.experts.<e>.{gate,up,down}_proj.{weight,scales,biases}, affine 4-bit g64) while the runtime module
is a stacked switch_mlp. The loader used to drop those keys (load_weights(strict=False)), leaving the switch at its
random float32 init (measured: cosine 3e-4 vs the bundle tensors; live MTP acceptance 21-27 % vs 59 % offline).
Pinned here: the stacked path is marked quantized with the experts' own layout, tensors are stacked in NUMERIC expert
order, and missing experts / mixed layouts / shape mismatches fail closed.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from vmlx_engine.utils import jang_loader as jl

E, D, I = 12, 128, 64


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.switch_mlp = SwitchGLU(D, I, E)


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.mtp = nn.Module()
        self.language_model.mtp.layers = [_Head()]


def _bundle_tensors(bits=4, gs=64, seed=0):
    """Per-expert quantized triplets as a JANG writer stores them (keys after the MTP sanitize rename)."""
    mx.random.seed(seed)
    out, dense = {}, {}
    for proj, (o, i) in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
        for e in range(E):
            w = mx.random.normal((o, i)) * 0.05
            q, s, b = mx.quantize(w, group_size=gs, bits=bits)
            base = f"language_model.mtp.layers.0.mlp.experts.{e}.{proj}"
            out[f"{base}.weight"], out[f"{base}.scales"], out[f"{base}.biases"] = q, s, b
            dense[(proj, e)] = mx.dequantize(q, s, b, group_size=gs, bits=bits)
    return out, dense


def test_switch_paths_and_override():
    paths = {f"mtp.layers.0.mlp.experts.{e}.gate_proj" for e in range(E)} | {"mtp.fc"}
    sw = jl._split_expert_switch_paths(paths)
    assert sw == {"mtp.layers.0.mlp.switch_mlp.gate_proj": set(range(E))}
    q = {f"mtp.layers.0.mlp.experts.{e}.gate_proj": {"bits": 4, "group_size": 64, "mode": "affine"} for e in range(E)}
    assert jl._split_expert_switch_override("mtp.layers.0.mlp.switch_mlp.gate_proj", set(range(E)), q) == {
        "bits": 4, "group_size": 64}
    q["mtp.layers.0.mlp.experts.3.gate_proj"] = {"bits": 8, "group_size": 64}
    with pytest.raises(ValueError):
        jl._split_expert_switch_override("mtp.layers.0.mlp.switch_mlp.gate_proj", set(range(E)), q)
    del q["mtp.layers.0.mlp.experts.3.gate_proj"]
    assert jl._split_expert_switch_override("mtp.layers.0.mlp.switch_mlp.gate_proj", set(range(E)), q) is None


def _quantized_root(bits=4, gs=64):
    root = _Root()
    nn.quantize(root, group_size=gs, bits=bits,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and ".switch_mlp." in p)
    return root


def test_stacks_in_numeric_order_and_loads():
    root = _quantized_root()
    tensors, dense = _bundle_tensors()
    switch_paths = {p for p, _ in root.named_modules() if p.endswith(("switch_mlp.gate_proj", "switch_mlp.up_proj",
                                                                      "switch_mlp.down_proj"))}
    assert len(switch_paths) == 3
    buffer = {}
    # feed in two "shards" with a scrambled key order (lexicographic 0, 1, 10, 11, 2, ... would mis-order a naive sort)
    keys = sorted(tensors)
    shard_a = {k: tensors[k] for k in keys[::2]}
    shard_b = {k: tensors[k] for k in keys[1::2]}
    shard_a["language_model.mtp.fc.weight"] = mx.zeros((2, 2))
    assert jl._take_split_expert_tensors(shard_a, switch_paths, buffer) == len(keys[::2])
    assert set(shard_a) == {"language_model.mtp.fc.weight"}           # unrelated keys stay
    jl._take_split_expert_tensors(shard_b, switch_paths, buffer)
    n = jl._load_split_expert_stacks(root, buffer)
    assert n == 9
    sw = root.language_model.mtp.layers[0].mlp.switch_mlp
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(sw, proj)
        for e in (0, 1, 2, 10, 11):
            got = mx.dequantize(lin["weight"][e], lin["scales"][e], lin["biases"][e],
                                group_size=lin.group_size, bits=lin.bits)
            assert mx.array_equal(got, dense[(proj, e)]), (proj, e)


def test_missing_expert_fails_closed():
    root = _quantized_root()
    tensors, _ = _bundle_tensors()
    tensors.pop("language_model.mtp.layers.0.mlp.experts.7.up_proj.scales")
    switch_paths = {p for p, _ in root.named_modules() if ".switch_mlp." in p}
    buffer = {}
    jl._take_split_expert_tensors(tensors, switch_paths, buffer)
    with pytest.raises(RuntimeError, match="missing"):
        jl._load_split_expert_stacks(root, buffer)


def test_layout_mismatch_fails_closed():
    root = _quantized_root(bits=8)                                   # module built 8-bit, bundle stores 4-bit
    tensors, _ = _bundle_tensors(bits=4)
    switch_paths = {p for p, _ in root.named_modules() if ".switch_mlp." in p}
    buffer = {}
    jl._take_split_expert_tensors(tensors, switch_paths, buffer)
    with pytest.raises(RuntimeError, match="!= module"):
        jl._load_split_expert_stacks(root, buffer)


def test_no_split_keys_is_a_noop():
    root = _quantized_root()
    assert jl._load_split_expert_stacks(root, {}) == 0

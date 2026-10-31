# SPDX-License-Identifier: Apache-2.0
"""Load the official MiniMax-M3.1 checkpoint as it is stored.

The release keeps every linear in MXFP8 (e4m3 weights, one e8m0 scale per 32
inputs, named ``weight_scale_inv``) and the routed experts in NVFP4 (two e2m1
codes per byte in ``weight_packed``, one e4m3 scale per 16 inputs in
``weight_scale``, and one ``weight_global_scale`` per expert matrix that the
weight is divided by). The router, norms, embeddings, lm_head and the vision
stack are unquantized.

MLX has native kernels for both formats, so nothing is dequantized: the stored
bytes are reinterpreted and the modules are quantized to match. ``gather_qmm``
takes no global scale, so each expert's output is multiplied by
``1 / weight_global_scale``. ``input_global_scale`` (an activation scale for
4-bit-activation serving) is not used: activations stay in the model dtype.

Bit-exactness of both kernels against a spec dequant of the real weights, and
of the per-expert output scaling through ``gather_qmm``, is recorded in the
MiniMax-M3.1 campaign log (stage 1a).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

EXPERT_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


class NVFP4SwitchLinear(QuantizedSwitchLinear):
    """Routed experts in NVFP4 with one global scale per expert matrix."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__(input_dims, output_dims, num_experts, bias=False, group_size=16, bits=4, mode="nvfp4")
        self.inv_global_scale = mx.ones((num_experts,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        y = super().__call__(x, indices, sorted_indices=sorted_indices)
        s = self["inv_global_scale"][indices]
        return (y * s.reshape(*s.shape, 1, 1)).astype(y.dtype)


def is_source_checkpoint(weights: dict) -> bool:
    return any(k.endswith((".weight_scale_inv", ".weight_packed")) for k in weights)


def mxfp8_pairs_to_mlx(weights: dict) -> set:
    """In place: every ``X.weight`` + ``X.weight_scale_inv`` pair becomes ``X.weight`` (the e4m3 bytes viewed as
    uint32) + ``X.scales`` (MLX mxfp8 layout, same bytes). Returns the module paths X."""
    mxfp8 = set()
    for k in [k for k in weights if k.endswith(".weight_scale_inv")]:
        base = k[: -len(".weight_scale_inv")]
        weights[base + ".scales"] = weights.pop(k)
        weights[base + ".weight"] = mx.view(weights[base + ".weight"], mx.uint32)
        mxfp8.add(base)
    return mxfp8


def convert_mxfp8_linears(model: nn.Module, weights: dict) -> None:
    """In place: every ``X.weight`` + ``X.weight_scale_inv`` pair becomes ``X.weight`` (the e4m3 bytes viewed as
    uint32) + ``X.scales``, and those modules of ``model`` become mxfp8 QuantizedLinear."""
    mxfp8 = mxfp8_pairs_to_mlx(weights)
    nn.quantize(model, class_predicate=lambda path, m: MXFP8 if path in mxfp8 else False)


def convert(model: nn.Module, weights: dict) -> dict:
    """Rewrite sanitized checkpoint weights into MLX quantized layouts and
    quantize ``model``'s modules to match. ``weights`` uses the model's names
    (``model.layers.N.mlp.experts.E.w1.weight_packed`` etc.)."""
    out = dict(weights)
    convert_mxfp8_linears(model, out)

    experts: dict = {}
    for k in [k for k in out if ".mlp.experts." in k]:
        v = out.pop(k)
        prefix, rest = k.split(".mlp.experts.")
        e, wn, field = rest.split(".")
        experts.setdefault((prefix, wn), {}).setdefault(int(e), {})[field] = v
    for (prefix, wn), per in experts.items():
        n = len(per)
        if sorted(per) != list(range(n)):
            raise ValueError(f"{prefix}: expert ids are not 0..{n - 1}")
        base = f"{prefix}.mlp.switch_mlp.{EXPERT_PROJ[wn]}"
        out[base + ".weight"] = mx.stack([mx.view(per[e]["weight_packed"], mx.uint32) for e in range(n)])
        out[base + ".scales"] = mx.stack([per[e]["weight_scale"] for e in range(n)])
        g = mx.stack([per[e]["weight_global_scale"].reshape(()) for e in range(n)]).astype(mx.float32)
        out[base + ".inv_global_scale"] = 1.0 / g

    for layer in model.model.layers:
        sw = getattr(layer.mlp, "switch_mlp", None)
        if sw is None:
            continue
        for name in EXPERT_PROJ.values():
            num_experts, output_dims, input_dims = getattr(sw, name).weight.shape
            setattr(sw, name, NVFP4SwitchLinear(input_dims, output_dims, num_experts))
    return out

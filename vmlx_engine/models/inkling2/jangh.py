"""Apply Inkling's two soft caps between packed JANGH expert projections."""

import mlx.core as mx
import mlx.nn as nn

from ...jangh import kernels
from ...jangh.switch import rotate_rows



def is_affine_decode_projection(module):
    return (
        type(module) is nn.QuantizedLinear
        and module.mode == "affine"
        and module.bits == 8
        and module.group_size == 64
        and module.get("biases") is not None
        and "bias" not in module
    )


def affine_decode_rows(x, weight, scales, biases):
    """Indexed one-row GEMV preserves the affine reduction at every position."""
    options = {"transpose": True, "group_size": 64, "bits": 8, "mode": "affine"}
    length = x.shape[1]
    if length == 1:
        return mx.quantized_matmul(x, weight, scales, biases, **options)
    positions = mx.arange(length, dtype=mx.uint32)
    return mx.gather_qmm(
        x.reshape(length, 1, x.shape[-1]),
        weight[None], scales[None], biases[None],
        lhs_indices=positions, rhs_indices=mx.zeros((length,), dtype=mx.uint32),
        **options,
    ).reshape(1, length, weight.shape[0])


class _AffineAttentionPack:
    def __init__(self, modules):
        self.sizes = tuple(module.scales.shape[0] for module in modules)
        names = ("weight", "scales", "biases")
        self.weights = tuple(
            mx.concatenate([getattr(module, name) for module in modules], axis=0)
            for name in names
        )
        mx.eval(*self.weights)
        offset = 0
        # The individual modules remain valid for prefill or distinct K/V
        # inputs. Each uses a view of the packed storage.
        for module, size in zip(modules, self.sizes):
            for name, weight in zip(names, self.weights):
                setattr(module, name, weight[offset:offset + size])
            offset += size

    def project(self, x, *, include_kv):
        sizes = self.sizes if include_kv else self.sizes[:3]
        width = sum(sizes)
        weight, scales, biases = (value[:width] for value in self.weights)
        output = affine_decode_rows(x, weight, scales, biases)
        cuts = [sum(sizes[:index]) for index in range(1, len(sizes))]
        return mx.split(output, cuts, axis=-1)


def prepare_jangh_decode(model):
    for layer in model.language_model.model.layers:
        attention = layer.self_attn
        modules = (
            attention.q_proj, attention.r_proj, attention.gate_proj,
            attention.k_proj, attention.v_proj,
        )
        if all(is_affine_decode_projection(module) for module in modules):
            attention._affine_decode_pack = _AffineAttentionPack(modules)
        for controller in getattr(layer, "controllers", {}).values():
            forget, gain = controller.forget_weight, controller.gain_weight
            if forget.shape != (2304, 64) or gain.shape != forget.shape or gain.dtype != forget.dtype:
                continue
            weights = mx.concatenate([forget, gain], axis=0)
            mx.eval(weights)
            controller._verify_gate_weights = weights
            controller.forget_weight, controller.gain_weight = mx.split(weights, 2, axis=0)


def softcapped_experts(switch, x, indices, activation):
    leading_shape = x.shape[:-1]
    hidden_size = x.shape[-1]
    top_k = indices.shape[-1]
    rows = x.reshape(-1, hidden_size)
    routes = indices.reshape(-1).astype(mx.uint32)
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj

    if switch._use_sorted(routes.size, top_k):
        order = mx.argsort(routes)
        inverse = mx.argsort(order)
        sorted_routes = routes[order]
        inputs = rotate_rows(rows, gate)[order // top_k]
        gate_values = kernels.gather_qmm_sorted(
            inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, sorted_routes, gate.bits
        )
        up_values = kernels.gather_qmm_sorted(
            inputs, up.tq2_packed, up.tq2_scales, up._cb, sorted_routes, up.bits
        )
        middle = activation(up_values.astype(x.dtype), gate_values.astype(x.dtype))
        output = kernels.gather_qmm_sorted(
            rotate_rows(middle, down),
            down.tq2_packed, down.tq2_scales, down._cb, sorted_routes, down.bits,
        )[inverse]
    else:
        rotate = gate.rotated and switch._rot_in_kernel()
        inputs = rows if rotate else rotate_rows(rows, gate)
        slot_fastest = rows.shape[0] > 1
        if (
            gate.bits == up.bits
            and gate.tq2_packed.shape == up.tq2_packed.shape
            and gate.tq2_scales.shape == up.tq2_scales.shape
        ):
            gate_values, up_values = kernels.gather_qmv(
                inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, routes, gate.bits,
                x_per_dispatch=False, packed_u=up.tq2_packed, scales_u=up.tq2_scales,
                rotate=rotate, slot_fastest=slot_fastest, return_pair=True,
            )
        else:
            gate_values = kernels.gather_qmv(
                inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, routes, gate.bits,
                x_per_dispatch=False, rotate=rotate, slot_fastest=slot_fastest,
            )
            up_values = kernels.gather_qmv(
                inputs, up.tq2_packed, up.tq2_scales, up._cb, routes, up.bits,
                x_per_dispatch=False, rotate=rotate, slot_fastest=slot_fastest,
            )
        middle = activation(up_values.astype(x.dtype), gate_values.astype(x.dtype))
        middle, rotate_down = switch._down_in(middle)
        output = kernels.gather_qmv(
            middle, down.tq2_packed, down.tq2_scales, down._cb, routes, down.bits,
            x_per_dispatch=True, rotate=rotate_down, slot_fastest=slot_fastest,
        )

    return output.astype(x.dtype).reshape(*leading_shape, top_k, hidden_size)

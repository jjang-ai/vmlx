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
    """Multi-row affine 8-bit g=64 GEMV: reads weights ONCE with identical per-row reduction."""
    length = x.shape[1]
    if (
        1 <= length <= 16
        and x.shape[0] == 1
        and x.shape[-1] % 256 == 0
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        from .kernels import affine8_g64_multi_gemv

        fused = affine8_g64_multi_gemv(x, weight, scales, biases, out_dtype=x.dtype)
        if fused is not None:
            return fused
    options = {"transpose": True, "group_size": 64, "bits": 8, "mode": "affine"}
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
            if controller.prepare_fused_decode():
                controller._verify_gate_weights = controller._fused_decode_pair[0]
                continue
            forget, gain = controller.forget_weight, controller.gain_weight
            if forget.shape != (2304, 64) or gain.shape != forget.shape or gain.dtype != forget.dtype:
                continue
            weights = mx.concatenate([forget, gain], axis=0)
            mx.eval(weights)
            controller._verify_gate_weights = weights
            controller.forget_weight, controller.gain_weight = mx.split(weights, 2, axis=0)


def softcapped_experts_routed(switch, x, indices, weights, activation):
    """Fused JANGH routed MoE for Inkling 2: 1-pass gate+up+softcapped-SwiGLU + fused weighted down."""
    leading_shape = x.shape[:-1]
    hidden_size = x.shape[-1]
    top_k = indices.shape[-1]
    rows = x.reshape(-1, hidden_size)
    idx2 = indices.reshape(-1, top_k).astype(mx.uint32)
    w2 = weights.reshape(-1, top_k).astype(mx.float32)
    routes = idx2.reshape(-1)
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    caps = (float(activation.gate_cap), float(activation.up_cap))
    same_gu = (
        gate.bits == up.bits
        and gate.tq2_packed.shape == up.tq2_packed.shape
        and gate.tq2_scales.shape == up.tq2_scales.shape
    )

    # Keep T=1..16 speculative verify (<= 128 routes, top_k=8) on row-exact QMV decode kernels:
    # on Inkling-2-Mini JANGH2, QMV is 2.0x faster than NAX sorted at verify widths (0.55 vs 1.13 ms/layer!)
    # AND guarantees bit-exact greedy parity between AR (L=1) and verify (L=2..16)!
    use_sorted = routes.size > 128 and switch._use_sorted(routes.size, top_k)

    if use_sorted:
        order = mx.argsort(routes)
        inverse = mx.argsort(order)
        sorted_routes = routes[order]
        inputs = rotate_rows(rows, gate)[order // top_k]
        use_nax = kernels.nax_available()
        plan = kernels.expert_tile_plan(sorted_routes, gate.num_experts) if use_nax else None
        if same_gu and plan is not None:
            middle = kernels.gather_qmm_expert_sorted(
                inputs, gate.tq2_packed, gate.tq2_scales, sorted_routes, gate.bits, plan,
                packed_u=up.tq2_packed, scales_u=up.tq2_scales, act="inkling", alpha=caps,
            )
        elif same_gu:
            middle = kernels.gather_qmm_sorted(
                inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, sorted_routes, gate.bits,
                packed_u=up.tq2_packed, scales_u=up.tq2_scales, act="inkling", alpha=caps,
            )
        else:
            gate_values = kernels.gather_qmm_sorted(
                inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, sorted_routes, gate.bits
            )
            up_values = kernels.gather_qmm_sorted(
                inputs, up.tq2_packed, up.tq2_scales, up._cb, sorted_routes, up.bits
            )
            middle = activation(up_values.astype(x.dtype), gate_values.astype(x.dtype))
        down_in = rotate_rows(middle, down)
        if plan is not None:
            down_sorted = kernels.gather_qmm_expert_sorted(
                down_in, down.tq2_packed, down.tq2_scales, sorted_routes, down.bits, plan
            )
        else:
            down_sorted = kernels.gather_qmm_sorted(
                down_in, down.tq2_packed, down.tq2_scales, down._cb, sorted_routes, down.bits
            )
        fused_out = kernels.weighted_unsort(down_sorted, inverse, w2, x.dtype)
        if fused_out is not None:
            return fused_out.reshape(*leading_shape, hidden_size)
        raw = down_sorted[inverse].reshape(rows.shape[0], top_k, hidden_size)
        return (raw.astype(mx.float32) * w2[..., None]).sum(axis=1).astype(x.dtype).reshape(*leading_shape, hidden_size)

    slot_fastest = rows.shape[0] > 1
    # Rotate T token rows ONCE on host instead of T*top_k times inside the kernel.
    inputs = kernels.h32_rows(rows, mx.float32) if gate.rotated else rows
    if same_gu:
        middle = kernels.gather_qmv(
            inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, routes, gate.bits,
            x_per_dispatch=False, packed_u=up.tq2_packed, scales_u=up.tq2_scales,
            rotate=False, slot_fastest=slot_fastest, act="inkling", alpha=caps,
        )
    else:
        gate_values = kernels.gather_qmv(
            inputs, gate.tq2_packed, gate.tq2_scales, gate._cb, routes, gate.bits,
            x_per_dispatch=False, rotate=False, slot_fastest=slot_fastest,
        )
        up_values = kernels.gather_qmv(
            inputs, up.tq2_packed, up.tq2_scales, up._cb, routes, up.bits,
            x_per_dispatch=False, rotate=False, slot_fastest=slot_fastest,
        )
        middle = activation(up_values.astype(x.dtype), gate_values.astype(x.dtype))
    down_in = kernels.h32_rows(middle, mx.float32) if down.rotated else middle
    out = kernels.gather_qmv_weighted_down(
        down_in, down.tq2_packed, down.tq2_scales, down._cb, idx2, w2, down.bits,
        x.dtype, rotate=False, slot_fastest=slot_fastest,
    )
    return out.reshape(*leading_shape, hidden_size)


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

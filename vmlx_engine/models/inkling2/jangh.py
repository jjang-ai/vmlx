"""Apply Inkling's two soft caps between packed JANGH expert projections."""

import mlx.core as mx

from ...jangh import kernels
from ...jangh.switch import rotate_rows


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

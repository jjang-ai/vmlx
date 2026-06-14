"""Regression gate for the MiniMax-M3 affine-2 SwitchGLU fast path."""

import os
import sys

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m3_affine2_switch import (
    MiniMaxM3Affine2SwitchGLU,
    affine2_switchglu_decode,
    can_use_affine2_switchglu,
    _make_affine2_gather_pair,
)


def _quantized_switch_glu(input_dims=128, hidden_dims=64, experts=8, cls=SwitchGLU):
    switch = cls(input_dims, hidden_dims, experts, bias=False)
    switch.gate_proj = switch.gate_proj.to_quantized(group_size=32, bits=2, mode="affine")
    switch.up_proj = switch.up_proj.to_quantized(group_size=32, bits=2, mode="affine")
    switch.down_proj = switch.down_proj.to_quantized(group_size=32, bits=2, mode="affine")
    mx.eval(switch.parameters())
    return switch


def main():
    switch = _quantized_switch_glu()
    x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
    expected = switch(x, indices)
    actual = affine2_switchglu_decode(switch, x, indices, opt=4)
    mx.eval(expected, actual)

    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    max_abs = float(mx.max(mx.abs(actual - expected)))
    assert max_abs < 1e-4, max_abs

    idx = indices.reshape(indices.size).astype(mx.uint32)
    x_rows = mx.broadcast_to(x.reshape(1, x.shape[-1]), (indices.size, x.shape[-1]))
    pair = _make_affine2_gather_pair(
        switch.gate_proj.input_dims,
        switch.gate_proj.output_dims,
        switch.gate_proj.group_size,
        indices.size,
        4,
    )
    gate_pair, up_pair = pair(x_rows, switch.gate_proj, switch.up_proj, idx)
    gate_ref = switch.gate_proj(x.reshape(1, 1, 1, 1, x.shape[-1]), indices).reshape(indices.size, -1)
    up_ref = switch.up_proj(x.reshape(1, 1, 1, 1, x.shape[-1]), indices).reshape(indices.size, -1)
    mx.eval(gate_pair, up_pair, gate_ref, up_ref)
    pair_max = max(
        float(mx.max(mx.abs(gate_pair - gate_ref))),
        float(mx.max(mx.abs(up_pair - up_ref))),
    )
    assert pair_max < 1e-4, pair_max

    guarded = _quantized_switch_glu(cls=MiniMaxM3Affine2SwitchGLU)
    x2 = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)
    indices2 = mx.array([[[1, 3, 5, 7], [0, 2, 4, 6]]], dtype=mx.uint32)
    assert can_use_affine2_switchglu(guarded, x, indices)
    os.environ["VMLX_M3_AFFINE2_SWITCH"] = "0"
    try:
        assert not can_use_affine2_switchglu(guarded, x, indices)
    finally:
        os.environ.pop("VMLX_M3_AFFINE2_SWITCH", None)
    assert not can_use_affine2_switchglu(guarded, x2, indices2)
    y2 = guarded(x2, indices2)
    mx.eval(y2)
    assert y2.shape == (1, 2, 4, 128), y2.shape

    print(
        f"OK: affine2 SwitchGLU fast path matches baseline, "
        f"max_abs={max_abs:.6g}, pair_max={pair_max:.6g}; multi-token fallback holds"
    )



if __name__ == "__main__":
    main()

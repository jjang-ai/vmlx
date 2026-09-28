"""The optional Bonsai GDN recurrence tracks the existing FP32 update."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.qwen3_5 import language as qlang

from vmlx_engine.metal.bonsai_gdn_recur import step


class _Layer:
    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = 128
    head_v_dim = 128
    conv_dim = 10240
    key_dim = 2048
    conv_kernel_size = 4

    def __init__(self):
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, 4, groups=self.conv_dim, bias=False)
        self.A_log = mx.zeros((self.num_v_heads,), mx.float32)
        self.dt_bias = mx.zeros((self.num_v_heads,), mx.float32)


def _stock(layer, qkv, a, b, conv_state, ssm_state):
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv = mx.contiguous(conv_input[:, -3:])
    conv_out = nn.silu(layer.conv1d(conv_input))
    q, k, v = [
        t.reshape(1, 1, heads, dim)
        for t, heads, dim in zip(
            mx.split(conv_out, [layer.key_dim, 2 * layer.key_dim], -1),
            [16, 16, 48], [128, 128, 128],
        )
    ]
    inv_scale = 128**-0.5
    q = inv_scale**2 * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    y, next_state = qlang.gated_delta_update(
        q, k, v, a, b, layer.A_log, layer.dt_bias, ssm_state,
        None, use_kernel=True,
    )
    return y, next_conv, next_state


@pytest.mark.parametrize("random_state", [False, True])
def test_bonsai_gdn_recurrence_matches_stock(monkeypatch, random_state):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("requires an MLX Metal GPU")
    monkeypatch.setenv("VMLX_BONSAI_GDN_FP32", "1")
    mx.random.seed(28)
    layer = _Layer()
    qkv = mx.random.normal((1, 1, 10240), dtype=mx.float32) * 0.05
    a = mx.random.normal((1, 1, 48), dtype=mx.float32) * 0.05
    b = mx.random.normal((1, 1, 48), dtype=mx.float32) * 0.05
    conv_state = mx.random.normal((1, 3, 10240), dtype=mx.float32) * 0.05
    state = (
        mx.random.normal((1, 48, 128, 128), dtype=mx.float32) * 0.05
        if random_state else mx.zeros((1, 48, 128, 128), mx.float32)
    )
    expected = _stock(layer, qkv, a, b, conv_state, state)
    actual = step(layer, qkv, a, b, conv_state, state)
    assert actual is not None
    mx.eval(*expected, *actual)
    for want, got, tolerance in zip(expected, actual, (2e-5, 0.0, 2e-5)):
        error = float(mx.max(mx.abs(want - got)).item())
        assert error <= tolerance


def test_bonsai_gdn_recurrence_rejects_wrong_dtype(monkeypatch):
    monkeypatch.setenv("VMLX_BONSAI_GDN_FP32", "1")
    layer = _Layer()
    qkv = mx.zeros((1, 1, 10240), mx.float16)
    a = mx.zeros((1, 1, 48), mx.float32)
    b = mx.zeros((1, 1, 48), mx.float32)
    conv = mx.zeros((1, 3, 10240), mx.float32)
    state = mx.zeros((1, 48, 128, 128), mx.float32)
    assert step(layer, qkv, a, b, conv, state) is None

"""Exact Qwen down-tail component checks, not a full-model speed claim."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from vmlx_engine.metal import qwen4_exact_down as candidate
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope


def test_default_requested_with_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_EXACT_DOWN", raising=False)
    assert candidate.exact_down_requested()
    for value in ("0", "false", "off"):
        monkeypatch.setenv("VMLX_QWEN4_EXACT_DOWN", value)
        assert not candidate.exact_down_requested()
    monkeypatch.setenv("VMLX_QWEN4_EXACT_DOWN", "1")
    assert candidate.exact_down_requested()


def test_unqualified_device_declines_before_tensor_inspection(monkeypatch):
    monkeypatch.setattr(candidate, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(candidate, "_compatible_runtime", lambda: False)
    def forbidden(*args):
        pytest.fail("unqualified runtime inspected candidate tensors")
    monkeypatch.setattr(candidate, "_eligible", forbidden)
    assert candidate.qwen4_exact_down(None, None, None, None, enabled=True) is None


def projection(bits=3, group=64):
    mx.random.seed(813)
    p = QuantizedSwitchLinear(640, 2560, 12, bias=False, bits=bits, group_size=group)
    p.scales = p.scales.astype(mx.float16)
    p.biases = p.biases.astype(mx.float16)
    p.eval()
    mx.eval(p.parameters())
    return p


@pytest.mark.parametrize("bits,group", [(2, 32), (2, 64), (3, 64), (4, 64), (6, 64), (8, 64)])
def test_native_down_weight_reduce_words(bits, group):
    if not mx.metal.is_available() or not candidate._compatible_runtime():
        pytest.skip("requires actual MLX0.32.2 or0.32.3/M5 Max")
    p = projection(bits, group)
    ids = mx.array([[[9, 1, 11, 0, 7, 3, 10, 2, 5, 8]]], dtype=mx.uint32)
    scores = mx.array([[[.18, .12, .11, .1, .08, .07, .065, .06, .04, .025]]], dtype=mx.float16)
    for scale in (0.0, 0.2, 3.0):
        z = (mx.random.normal((1, 1, 10, 1, 640)) * scale).astype(mx.float16)
        expected = (p(z, ids).squeeze(-2) * scores[..., None]).sum(axis=-2)
        with affine_moe_ar_scope():
            got = candidate.qwen4_exact_down(p, z, ids, scores, enabled=True)
        assert got is not None
        mx.eval(got, expected)
        np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(expected).view(np.uint16))


def test_disabled_and_outside_ar_do_not_inspect_or_dispatch(monkeypatch):
    def forbidden(*args):
        pytest.fail("disabled or speculative path inspected candidate tensors")
    monkeypatch.setattr(candidate, "_eligible", forbidden)
    assert candidate.qwen4_exact_down(None, None, None, None, enabled=False) is None
    assert candidate.qwen4_exact_down(None, None, None, None, enabled=True) is None


@pytest.mark.parametrize("bits,group", [(2, 32), (2, 64), (3, 64), (4, 64), (6, 64), (8, 64)])
def test_verify_rows_down_matches_native_projection(bits, group):
    if not mx.metal.is_available() or not candidate._compatible_runtime():
        pytest.skip("requires qualified MLX / M5 Max")
    from vmlx_engine.metal.qwen4_rows_exact_moe import _down_rows_kernel
    p = projection(bits, group)
    ids = mx.array([[[9, 1, 11, 0, 7, 3, 10, 2, 5, 8],
                     [0, 2, 4, 6, 8, 10, 1, 3, 5, 7]]], dtype=mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 2, 10)), axis=-1).astype(mx.float16)
    z = (mx.random.normal((1, 2, 10, 1, 640)) * .2).astype(mx.float16)
    expected = mx.concatenate([
        (p(z[:, i:i+1], ids[:, i:i+1]).squeeze(-2) * scores[:, i:i+1, :, None]).sum(axis=-2)
        for i in range(2)
    ], axis=1)
    got = _down_rows_kernel()(
        inputs=[z, p.weight, p.scales, p.biases, ids, scores],
        template=[("T", mx.float16), ("BITS", bits), ("GS", group),
                  ("EXPERTS", p.weight.shape[0]), ("ROWS", 8)],
        grid=(320 * 320, 2, 1), threadgroup=(320, 1, 1),
        output_shapes=[(1, 2, 2560)], output_dtypes=[mx.float16])[0]
    mx.eval(got, expected)
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(expected).view(np.uint16))


def test_rejects_changed_shape_dtype_and_projection():
    p = projection()
    z = mx.zeros((1, 1, 10, 1, 640), dtype=mx.float16)
    ids = mx.arange(10, dtype=mx.uint32).reshape(1, 1, 10)
    scores = mx.ones((1, 1, 10), dtype=mx.float16)
    assert candidate._eligible(p, z, ids, scores)
    assert not candidate._eligible(p, z, ids, scores.astype(mx.float32))
    assert not candidate._eligible(p, z.astype(mx.bfloat16), ids, scores)
    assert not candidate._eligible(p, mx.zeros((1, 2, 10, 1, 640), dtype=mx.float16), ids, scores)
    p.train()
    assert not candidate._eligible(p, z, ids, scores)
    p.eval()
    p.bias = mx.zeros((12, 2560), dtype=mx.float16)
    assert not candidate._eligible(p, z, ids, scores)


def test_invalid_internal_id_cannot_read_outside_weight_bank():
    if not mx.metal.is_available() or not candidate._compatible_runtime():
        pytest.skip("requires actual MLX0.32.2 or0.32.3/M5 Max")
    p = projection()
    z = mx.ones((1, 1, 10, 1, 640), dtype=mx.float16)
    ids = mx.array([[[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]]], dtype=mx.int32)
    scores = mx.ones((1, 1, 10), dtype=mx.float16)
    with affine_moe_ar_scope():
        got = candidate.qwen4_exact_down(p, z, ids, scores, enabled=True)
    assert got is not None
    # Invalid model-internal data is exposed as nonfinite, never a valid-looking
    # invented expert output. Production IDs come directly from argpartition.
    assert bool(mx.all(mx.isnan(got)))


@pytest.mark.parametrize("version,device,expected", [
    ("0.32.2", "Apple M5 Max", True),
    ("0.32.3", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M4 Max", False),
    ("0.32.4", "Apple M5 Max", False),
    ("0.32.3", "Apple M5 Ultra", False),
    ("0.32.3", "Apple M4 Max", False),
])
def test_qualified_runtime_policy(monkeypatch, version, device, expected):
    candidate._compatible_runtime.cache_clear()
    monkeypatch.setattr(candidate.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    try:
        assert candidate._compatible_runtime() is expected
    finally:
        candidate._compatible_runtime.cache_clear()

"""Narrow HC combine/norm admission plus exact native primitive parity."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal import qwen4_hc_norm as hc
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope


def metadata():
    return [SimpleNamespace(shape=shape, dtype=mx.float16) for shape in
            ((1, 1, 10240), (1, 1, 2560), (1, 1, 4), (10240,))]


def test_default_requested_with_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HC_COMBINE_NORM", raising=False)
    assert hc.hc_combine_norm_requested()
    monkeypatch.setenv("VMLX_QWEN4_HC_COMBINE_NORM", "0")
    assert not hc.hc_combine_norm_requested()
    monkeypatch.setenv("VMLX_QWEN4_HC_COMBINE_NORM", "1")
    assert hc.hc_combine_norm_requested()


@pytest.mark.parametrize("index,shape", [
    (0, (2, 1, 10240)), (0, (1, 2, 10240)), (0, (1, 1, 5120)),
    (1, (1, 1, 1280)), (2, (1, 1, 8)), (3, (4, 2560)),
])
def test_shape_guards(index, shape):
    args = metadata()
    assert hc.hc_combine_norm_eligible(*args, eps=1e-6, group_size=2560)
    args[index].shape = shape
    assert not hc.hc_combine_norm_eligible(*args, eps=1e-6, group_size=2560)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_no_dtype_promotion(index, dtype):
    args = metadata()
    args[index].dtype = dtype
    assert not hc.hc_combine_norm_eligible(*args, eps=1e-6, group_size=2560)


@pytest.mark.parametrize("eps", [-1, float("nan"), float("inf"), None])
def test_invalid_epsilon(eps):
    assert not hc.hc_combine_norm_eligible(*metadata(), eps=eps, group_size=2560)


def test_wrong_group_size():
    assert not hc.hc_combine_norm_eligible(*metadata(), eps=1e-6, group_size=1280)


def test_disabled_and_outside_ar_never_construct_kernel(monkeypatch):
    def forbidden():
        pytest.fail("ineligible call constructed a kernel")
    monkeypatch.setattr(hc, "_kernel", forbidden)
    assert hc.hc_combine_norm(*metadata(), eps=1e-6, group_size=2560, enabled=False) is None
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: False)
    assert hc.hc_combine_norm(*metadata(), eps=1e-6, group_size=2560, enabled=True) is None


def test_runtime_mismatch_never_constructs_kernel(monkeypatch):
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(hc, "_compatible_runtime", lambda: False)
    def forbidden():
        pytest.fail("unsupported runtime constructed a kernel")
    monkeypatch.setattr(hc, "_kernel", forbidden)
    assert hc.hc_combine_norm(*metadata(), eps=1e-6, group_size=2560, enabled=True) is None


def test_admitted_kernel_failure_is_not_replayed(monkeypatch):
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(hc, "_compatible_runtime", lambda: True)
    monkeypatch.setattr(hc, "_epsilon", lambda _: object())
    monkeypatch.setattr(hc, "_axis_size", lambda: object())
    failure = RuntimeError("admitted kernel failure")
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise failure
    monkeypatch.setattr(hc, "_kernel", lambda: fail)
    with pytest.raises(RuntimeError) as caught:
        hc.hc_combine_norm(*metadata(), eps=1e-6, group_size=2560, enabled=True)
    assert caught.value is failure and len(calls) == 1


def test_decoder_hook_keeps_unqualified_paths_on_original_route():
    source = (Path(__file__).parents[1] /
              "vmlx_engine/models/qwen4_exp/language.py").read_text()
    begin = source.index("        combined_norm = None", source.index("class DecoderLayer"))
    end = source.index("        r = self.mlp(x,", begin)
    hook = source[begin:end]
    for guard in ("profile_layer is None", "mask is None", "not self.training",
                  "not n_confirmed", "not prefill_checkpoint_steps", "not last_token_only"):
        assert guard in hook
    assert "self.attn_hyper_connection.combine(hyper, r, inject)" in hook
    assert "self.mlp_hyper_connection(h)" in hook
    assert "from_normed(h, combined_norm[1])" in hook


@pytest.mark.parametrize("pattern", ["random", "cancellation", "small", "zero"])
def test_native_fp16_primitive_bit_parity(pattern):
    # Root-only execution on the pinned target. No dtype/runtime substitution.
    if not mx.metal.is_available() or not hc._compatible_runtime():
        pytest.skip("requires qualified MLX 0.32.2 or 0.32.3 / M5 Max")
    rng = np.random.default_rng(49231)
    residual = rng.normal(size=(1, 1, 10240)).astype(np.float16)
    block = rng.normal(size=(1, 1, 2560)).astype(np.float16)
    inject = np.array([[[0.001, 0.5, 1.0, 1.999]]], dtype=np.float16)
    weight = rng.uniform(-1.5, 1.5, size=10240).astype(np.float16)
    if pattern == "cancellation":
        residual = -(block[..., None, :] * inject[..., :, None]).reshape(residual.shape)
        residual[..., ::2] = np.nextafter(residual[..., ::2], np.float16(np.inf))
    elif pattern == "small":
        residual *= np.float16(2**-12)
        block *= np.float16(2**-12)
    elif pattern == "zero":
        residual.fill(0)
        block.fill(0)
    residual, block, inject, weight = (mx.array(x) for x in (residual, block, inject, weight))
    combined = (residual + (block[..., None, :] * inject[..., :, None]).reshape(residual.shape)).astype(mx.float16)
    normalized = mx.fast.rms_norm(combined.reshape(1, 1, 4, 2560), None, 1e-6)
    expected = normalized.reshape(1, 1, 10240) * weight
    with affine_moe_ar_scope():
        actual = hc.hc_combine_norm(residual, block, inject, weight,
                                    eps=1e-6, group_size=2560, enabled=True)
    assert actual is not None
    mx.eval(combined, expected, *actual)
    for got, want in zip(actual, (combined, expected)):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16))


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
    hc._compatible_runtime.cache_clear()
    monkeypatch.setattr(hc.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    try:
        assert hc._compatible_runtime() is expected
    finally:
        hc._compatible_runtime.cache_clear()

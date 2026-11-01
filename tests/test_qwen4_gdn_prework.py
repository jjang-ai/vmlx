"""Q32 admission and root-only exact primitive/unchanged-recurrence screen."""

from types import SimpleNamespace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vmlx_engine.metal import qwen4_gdn_prework as pre


def metadata():
    shapes = [(1, 1, 10240), (1, 1, 48), (1, 1, 48), (1, 3, 10240),
              (10240, 4, 1), (48,), (48,)]
    return [SimpleNamespace(shape=s, dtype=mx.float16 if i < 5 else mx.float32)
            for i, s in enumerate(shapes)]


def test_default_requested_with_explicit_opt_out(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_GDN_PREWORK", raising=False)
    assert pre.gdn_prework_requested()
    monkeypatch.setenv("VMLX_QWEN4_GDN_PREWORK", "0")
    assert not pre.gdn_prework_requested()
    monkeypatch.setenv("VMLX_QWEN4_GDN_PREWORK", "1")
    assert pre.gdn_prework_requested()


@pytest.mark.parametrize("index", range(7))
def test_shapes(index):
    args = metadata()
    assert pre.gdn_prework_eligible(*args)
    args[index].shape = (2, *args[index].shape[1:])
    assert not pre.gdn_prework_eligible(*args)


@pytest.mark.parametrize("index", range(5))
def test_dtype_no_coercion(index):
    args = metadata()
    args[index].dtype = mx.bfloat16
    assert not pre.gdn_prework_eligible(*args)


@pytest.mark.parametrize("a_dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("dt_dtype", [mx.bfloat16, mx.float32])
def test_coefficients_preserved_independently(a_dtype, dt_dtype):
    args = metadata()
    args[5].dtype, args[6].dtype = a_dtype, dt_dtype
    assert pre.gdn_prework_eligible(*args)
    assert mx.result_type(mx.float16, dt_dtype) == mx.float32
    assert args[5].dtype == a_dtype and args[6].dtype == dt_dtype


@pytest.mark.parametrize("index", [5, 6])
@pytest.mark.parametrize("dtype", [mx.float16, mx.int32])
def test_unsupported_coefficients(index, dtype):
    args = metadata()
    args[index].dtype = dtype
    assert not pre.gdn_prework_eligible(*args)


@pytest.mark.parametrize("option", ["mask", "lengths", "training", "incumbent_fused_conv"])
def test_other_routes_excluded(option):
    assert not pre.gdn_prework_eligible(*metadata(), **{option: True})


def test_disabled_scope_and_unsupported_runtime_do_not_dispatch(monkeypatch):
    def forbidden():
        pytest.fail("ineligible call constructed a kernel")
    monkeypatch.setattr(pre, "_kernel", forbidden)
    assert pre.gdn_prework(*metadata(), enabled=False) is None
    monkeypatch.setattr(pre, "affine_moe_ar_scope_active", lambda: False)
    assert pre.gdn_prework(*metadata(), enabled=True) is None
    monkeypatch.setattr(pre, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(pre, "_compatible_runtime", lambda: False)
    assert pre.gdn_prework(*metadata(), enabled=True) is None


def test_admitted_failure_propagates_without_retry(monkeypatch):
    monkeypatch.setattr(pre, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(pre, "_compatible_runtime", lambda: True)
    monkeypatch.setattr(pre, "_constants", lambda: ())
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("kernel failure")
    monkeypatch.setattr(pre, "_kernel", lambda: fail)
    with pytest.raises(RuntimeError, match="kernel failure"):
        pre.gdn_prework(*metadata(), enabled=True)
    assert len(calls) == 1


@pytest.mark.parametrize("state", [
    SimpleNamespace(shape=(2, 48, 128, 128), dtype=mx.float32),
    SimpleNamespace(shape=(1, 48, 128, 128), dtype=mx.float16),
])
def test_update_bad_state_declines_before_construction(monkeypatch, state):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported state constructed prework")
    monkeypatch.setattr(pre, "gdn_prework", forbidden)
    assert pre.gdn_prework_update(*metadata(), state, enabled=True) is None


def test_update_none_state_and_logging_are_graph_only(monkeypatch, caplog):
    import mlx_lm.models.gated_delta as gd
    values = tuple(object() for _ in range(6))
    zero, output, new_state = object(), object(), object()
    calls = []
    monkeypatch.setattr(pre, "gdn_prework", lambda *args, **kwargs: values)
    monkeypatch.setattr(pre, "_ENGAGEMENT_LOGGED", False)
    def zeros(shape, *, dtype):
        assert shape == (1, 48, 128, 128) and dtype == mx.float32
        return zero
    def recurrence(*args):
        calls.append(args)
        assert args == (*values[:3], values[4], values[5], zero)
        return output, new_state
    monkeypatch.setattr(mx, "zeros", zeros)
    monkeypatch.setattr(gd, "gated_delta_kernel", recurrence)
    monkeypatch.setattr(mx, "eval", lambda *args: pytest.fail("host evaluation"))
    with caplog.at_level("INFO"):
        for _ in range(2):
            got = pre.gdn_prework_update(*metadata(), None, enabled=True)
            assert got == (output, values[3], new_state)
    assert len(calls) == 2
    assert caplog.text.count("constructed ordinary-AR") == 1


def test_update_recurrence_failure_not_replayed(monkeypatch):
    import mlx_lm.models.gated_delta as gd
    monkeypatch.setattr(pre, "gdn_prework", lambda *args, **kwargs: (None,) * 6)
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError("native recurrence failure")
    monkeypatch.setattr(gd, "gated_delta_kernel", fail)
    state = SimpleNamespace(shape=(1, 48, 128, 128), dtype=mx.float32)
    with pytest.raises(RuntimeError, match="native recurrence failure"):
        pre.gdn_prework_update(*metadata(), state, enabled=True)
    assert len(calls) == 1


def test_hook_keeps_other_routes_unchanged():
    source = (Path(__file__).parents[1] / 'vmlx_engine/models/qwen4_exp/language.py').read_text()
    start = source.index('            preworked = gdn_prework_update(')
    end = source.index('\n        if cache is not None:', start)
    hook = source[start:end]
    for guard in ('self._gdn_prework', '(batch_size, seq_len) == (1, 1)',
                  'n_confirmed == 0', 'not prefill_checkpoint_steps',
                  'precise_unmasked', 'cache is not None', 'training=self.training',
                  'mask=mask', 'lengths=lengths', 'incumbent_fused_conv=self._fused_conv_decode'):
        assert guard in hook
    assert 'if preworked is None:' in hook
    assert 'self._process_chunk(' in hook
    assert 'out, conv_f, ssm_f = preworked' in hook


def reference(args):
    from mlx_lm.models.gated_delta import compute_g
    qkv, a, b, history, weight, A_log, dt_bias = args
    joined = mx.concatenate([history, qkv], axis=1)
    conv = nn.silu(mx.conv1d(joined, weight, groups=10240))
    q, k, v = mx.split(conv, [2048, 4096], axis=-1)
    q, k = [x.reshape(1, 1, 16, 128) for x in (q, k)]
    inv_scale = 128 ** -0.5
    return (inv_scale**2 * mx.fast.rms_norm(q, None, 1e-6),
            inv_scale * mx.fast.rms_norm(k, None, 1e-6),
            v.reshape(1, 1, 48, 128), mx.contiguous(joined[:, -3:, :]),
            compute_g(A_log, a, dt_bias), mx.sigmoid(b))


def words(x):
    host = np.asarray(x)
    return host.view(np.uint16 if x.dtype == mx.float16 else np.uint32)


@pytest.mark.parametrize("pattern,coefficient_dtype", [
    ("random", mx.float32), ("zero", mx.float32),
    ("small", mx.float32), ("gate_extremes", mx.float32),
    ("random", mx.bfloat16),
])
def test_six_outputs_and_unchanged_recurrence_exact(pattern, coefficient_dtype):
    if not mx.metal.is_available() or not pre._compatible_runtime():
        pytest.skip("requires actual MLX 0.32.2 or 0.32.3 / M5 Max")
    from mlx_lm.models.gated_delta import gated_delta_kernel
    from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
    rng = np.random.default_rng(49232)
    hosts = [rng.normal(0, .2, size=x.shape).astype(
        np.float16 if x.dtype == mx.float16 else np.float32) for x in metadata()]
    if pattern == "zero":
        for x in hosts[:5]:
            x.fill(0)
    elif pattern == "small":
        for x in hosts[:5]:
            x *= np.float16(2**-10)
    elif pattern == "gate_extremes":
        hosts[1][...] = np.linspace(-20, 20, 48).astype(np.float16)
        hosts[2][...] = np.linspace(-15, 15, 48).astype(np.float16)
        hosts[5][...] = np.linspace(-4, 4, 48)
        hosts[6][...] = np.linspace(-2, 2, 48)
    args = tuple(mx.array(x, dtype=coefficient_dtype if i >= 5 else mx.float16)
                 for i, x in enumerate(hosts))
    expected = reference(args)
    with affine_moe_ar_scope():
        actual = pre.gdn_prework(*args, enabled=True)
    assert actual is not None
    mx.eval(*expected, *actual)
    for name, a, b in zip(("q", "k", "v", "history", "g", "beta"), actual, expected):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(words(a), words(b), err_msg=name)
    state = mx.array(rng.normal(0, .02, (1, 48, 128, 128)).astype(np.float32))
    def recur(x):
        q, k, v, _, g, beta = x
        return gated_delta_kernel(q, k, v, g, beta, state)
    expected_step, actual_step = recur(expected), recur(actual)
    mx.eval(*expected_step, *actual_step)
    for a, b in zip(actual_step, expected_step):
        np.testing.assert_array_equal(words(a), words(b))


@pytest.mark.parametrize("version,device,expected", [
    ("0.32.2", "Apple M5 Max", True),
    ("0.32.3", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M4 Max", False),
    ("0.32.4", "Apple M5 Max", False),
    ("0.33.0", "Apple M5 Max", False),
    ("0.32.3", "Apple M5 Ultra", False),
    ("0.32.3", "Apple M4 Max", False),
])
def test_qualified_version_and_device(monkeypatch, version, device, expected):
    pre._compatible_runtime.cache_clear()
    monkeypatch.setattr(pre.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    try:
        assert pre._compatible_runtime() is expected
    finally:
        pre._compatible_runtime.cache_clear()


def test_version_source_preserves_old_math_and_rejects_future():
    assert pre._source_for_version("0.32.2") == pre._SOURCE
    assert pre._source_for_version("0.32.5.dev20261101+360078a8") == pre._source_for_version("0.32.3")
    with pytest.raises(ValueError, match="unqualified"):
        pre._source_for_version("0.32.4")

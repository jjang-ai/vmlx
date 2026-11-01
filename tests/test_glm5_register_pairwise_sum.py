# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 reduction/state and fail-closed dispatch for GLM bulk prefill."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vmlx_engine.models.glm5_next import kda
from vmlx_engine.metal import glm5_pairwise_sum as fused


def exact(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)))


@pytest.fixture
def qualified(monkeypatch):
    if not mx.metal.is_available() or not fused._compatible_runtime():
        pytest.skip("register-sum numerical qualification requires qualified MLX 0.32.2 or 0.32.3/M5")
    monkeypatch.setattr(fused, "_FAILED", False)
    monkeypatch.setattr(fused, "_OBSERVED", False)
    monkeypatch.setattr(kda, "_EXACT_PAIRWISE_REQUESTED", True)


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 64, 64, 128), (2, 2, 64, 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_exact_product_sum_bits(qualified, shape, strided):
    mx.random.seed(819)
    left, right, gates = [mx.random.normal(shape) for _ in range(3)]
    gates = mx.cumsum(-mx.sigmoid(gates), axis=-2)
    if strided:
        left = mx.stack((left, right), axis=2)[:, :, 0]
        right = mx.stack((right, left), axis=2)[:, :, 0]
        gates = mx.stack((gates, -gates), axis=2)[:, :, 0]
    expected = mx.sum(kda._exact_pairwise_product(left, right, gates), axis=-1)
    actual = fused.glm5_pairwise_sum(left, right, gates, enabled=True)
    assert actual is not None
    exact(expected, actual)


@pytest.mark.parametrize("case", ["signed_zero", "cancellation", "underflow", "tiny"])
def test_edge_arithmetic(qualified, case):
    shape = (1, 2, 64, 128)
    left, right, gates = mx.ones(shape), mx.ones(shape), mx.zeros(shape)
    if case == "signed_zero":
        left = mx.broadcast_to(mx.array([-0.0]), shape)
    elif case == "cancellation":
        left = mx.broadcast_to(mx.array([1.0e15, 1.0, -1.0e15, 1.0] * 32), shape)
    elif case == "underflow":
        gates = mx.broadcast_to(mx.linspace(0, -1.0e5, 64)[None, None, :, None], shape)
    else:
        left = mx.full(shape, 1.0e-20)
    expected = mx.sum(kda._exact_pairwise_product(left, right, gates), axis=-1)
    actual = fused.glm5_pairwise_sum(left, right, gates, enabled=True)
    exact(expected, actual)


@pytest.mark.parametrize("tokens", [63, 64, 65, 127, 257, 513])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_product_path_chunked_state_and_decode(qualified, monkeypatch, tokens, dtype):
    mx.random.seed(tokens)
    shape = (1, tokens, 2, 128)
    q = kda.l2norm(mx.random.normal(shape)).astype(dtype)
    k = kda.l2norm(mx.random.normal(shape)).astype(dtype)
    v = mx.random.normal(shape).astype(dtype)
    g = -mx.sigmoid(mx.random.normal(shape))
    beta = mx.sigmoid(mx.random.normal(shape[:-1]))
    initial = mx.random.normal((1, 2, 128, 128)) * .01
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", False)
    expected, se = kda.kda_chunked(q, k, v, g, beta, initial)
    mx.eval(expected, se)
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", True)
    actual, sa = kda.kda_chunked(q, k, v, g, beta, initial)
    exact(expected, actual)
    exact(se, sa)
    assert fused._OBSERVED, "test must execute the real candidate, not its fallback"
    for _ in range(3):
        x = mx.random.normal((1, 2, 128))
        args = (kda.l2norm(x), kda.l2norm(x), x, -mx.ones_like(x), mx.full((1, 2), .5))
        oe, se = kda.kda_step(*args, se)
        oa, sa = kda.kda_step(*args, sa)
        exact(oe, oa)
        exact(se, sa)


def test_unqualified_shapes_dtypes_and_disabled_never_launch(monkeypatch):
    monkeypatch.setattr(fused, "_FAILED", False)
    def forbidden():
        raise AssertionError("ineligible input reached kernel")
    monkeypatch.setattr(fused, "_kernel", forbidden)
    x = mx.zeros((1, 2, 64, 128))
    assert fused.glm5_pairwise_sum(x, x, x, enabled=False) is None
    for y in [x[:, :, :32], x[..., :64], x.astype(mx.bfloat16), x[0], x[:0]]:
        assert fused.glm5_pairwise_sum(y, y, y, enabled=True) is None
    assert fused.glm5_pairwise_sum(x, x[:, :1], x, enabled=True) is None
    monkeypatch.setattr(fused, "_compatible_runtime", lambda: False)
    assert fused.glm5_pairwise_sum(x, x, x, enabled=True) is None


def test_first_launch_failure_disables_candidate_and_preserves_stock(qualified, monkeypatch, caplog):
    def broken():
        raise RuntimeError("controlled shader compilation failure")
    monkeypatch.setattr(fused, "_kernel", broken)
    x = mx.ones((1, 2, 64, 128))
    assert fused.glm5_pairwise_sum(x, x, x, enabled=True) is None
    assert fused._FAILED
    assert "retaining stock reduction" in caplog.text
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", True)
    exact(kda._pairwise_sum(x, x, x), mx.full((1, 2, 64, 64), 128.0))


@pytest.mark.parametrize("qualified_runtime", [False, True])
def test_policy_qualified_default_and_namespace_isolation_glm_only(monkeypatch, qualified_runtime):
    from vmlx_engine import glm5_prefill_policy as policy
    from vmlx_engine.glm5_prefill_policy import glm5_register_pairwise_sum_requested
    from vmlx_engine.prefix_cache import compute_model_cache_key
    monkeypatch.setattr(policy, "_register_sum_default_qualified", lambda: qualified_runtime)
    monkeypatch.delenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", raising=False)
    assert glm5_register_pairwise_sum_requested() is qualified_runtime
    for family in ("glm5_next", "glm5_next_text", "qwen4_exp", "qwen3_5"):
        model = SimpleNamespace(args=SimpleNamespace(model_type=family))
        monkeypatch.delenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", raising=False)
        default = compute_model_cache_key(model)
        monkeypatch.setenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", "0")
        off = compute_model_cache_key(model)
        assert (off != default) == (qualified_runtime and family.startswith("glm5_next"))
        monkeypatch.setenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", "1")
        assert glm5_register_pairwise_sum_requested() is True
        on = compute_model_cache_key(model)
        assert (on != off) == family.startswith("glm5_next")
        assert default == (on if qualified_runtime else off)


@pytest.mark.parametrize("value", ["0", "false", "off", "", "auto", "2"])
def test_explicit_non_one_preserves_disable(monkeypatch, value):
    from vmlx_engine import glm5_prefill_policy as policy
    monkeypatch.setattr(policy, "_register_sum_default_qualified", lambda: True)
    monkeypatch.setenv("VMLX_GLM5_REGISTER_PAIRWISE_SUM", value)
    assert not policy.glm5_register_pairwise_sum_requested()


@pytest.mark.parametrize("device,version,available,expected", [
    ("Apple M5 Max", "0.32.2", True, True),
    ("Apple M5 Pro", "0.32.2", True, False),
    ("Apple M3 Ultra", "0.32.2", True, False),
    ("Apple M5 Max", "0.32.3", True, True),
    ("Apple M5 Max", "0.32.5.dev20261101+360078a8", True, True),
    ("Apple M5 Ultra", "0.32.5.dev20261101+360078a8", True, False),
    ("Apple M5 Max", "0.32.5.dev20261101+360078a8", False, False),
    ("Apple M5 Max", "0.32.4", True, False),
    ("Apple M5 Max", "0.33.0", True, False),
    ("Apple M5 Pro", "0.32.3", True, False),
    ("Apple M5 Max", "0.32.3", False, False),
    ("Apple M5 Max", "0.32.2", False, False),
])
def test_automatic_runtime_qualification(monkeypatch, device, version, available, expected):
    import importlib.metadata
    from vmlx_engine import glm5_prefill_policy as policy
    policy._register_sum_default_qualified.cache_clear()
    monkeypatch.setattr(importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    monkeypatch.setattr(mx.metal, "is_available", lambda: available)
    try:
        assert policy._register_sum_default_qualified() is expected
    finally:
        policy._register_sum_default_qualified.cache_clear()


def test_automatic_runtime_lookup_failure_keeps_stock(monkeypatch):
    from vmlx_engine import glm5_prefill_policy as policy
    policy._register_sum_default_qualified.cache_clear()
    def unavailable():
        raise RuntimeError("controlled unavailable GPU")
    monkeypatch.setattr(mx.metal, "is_available", unavailable)
    try:
        assert not policy._register_sum_default_qualified()
    finally:
        policy._register_sum_default_qualified.cache_clear()


@pytest.mark.parametrize("batch,explicit,expected", [(1, False, True), (2, False, False), (2, True, True)])
def test_automatic_selection_does_not_promote_batched_prefill(monkeypatch, batch, explicit, expected):
    observed = []
    def selection(left, right, gates, *, enabled):
        observed.append(enabled)
        return None
    monkeypatch.setattr(kda, "glm5_pairwise_sum", selection)
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_REQUESTED", True)
    monkeypatch.setattr(kda, "_REGISTER_PAIRWISE_EXPLICIT", explicit)
    monkeypatch.setattr(kda, "_EXACT_PAIRWISE_REQUESTED", False)
    x = mx.ones((batch, 1, 64, 128))
    # Fallback is the actual native expression, not an invented result.
    exact(kda._pairwise_sum(x, x, x), mx.full((batch, 1, 64, 64), 128.0))
    assert observed == [expected]


@pytest.mark.parametrize("version,device,expected", [
    ("0.32.2", "Apple M5 Max", True),
    ("0.32.3", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M5 Ultra", False),
    ("0.32.2", "Apple M5 Pro", True),
    ("0.32.3", "Apple M5 Pro", False),
    ("0.32.3", "Apple M5 Ultra", False),
    ("0.32.4", "Apple M5 Max", False),
    ("0.33.0", "Apple M5 Max", False),
    ("0.32.3", "Apple M4 Max", False),
])
def test_kernel_runtime_version_qualification(monkeypatch, version, device, expected):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    fused._compatible_runtime.cache_clear()
    try:
        assert fused._compatible_runtime() is expected
    finally:
        fused._compatible_runtime.cache_clear()

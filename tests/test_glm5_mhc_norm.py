"""Focused GLM compound normalization admission and numerical contracts."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.metal import glm5_mhc_norm as hc
from vmlx_engine.metal.glm5_mhc_decode import glm5_mhc_decode

KW = dict(rms_eps=1e-5, sink_eps=1e-6, norm_eps=1e-5, iterations=20)


def metadata():
    return [SimpleNamespace(shape=s, dtype=d) for s, d in (
        ((1, 1, 4, 4096), mx.bfloat16), ((24, 16384), mx.bfloat16),
        ((24,), mx.float32), ((3,), mx.float32), ((4096,), mx.bfloat16))]


@pytest.mark.parametrize("index,shape", [
    (0, (2, 1, 4, 4096)), (0, (1, 2, 4, 4096)), (0, (1, 1, 4, 2048)),
    (1, (24, 8192)), (2, (4, 6)), (3, (1, 3)), (4, (1, 4096)),
])
def test_shape_admission(index, shape):
    args = metadata()
    assert hc.eligible(*args, **KW)
    args[index].shape = shape
    assert not hc.eligible(*args, **KW)


@pytest.mark.parametrize("index", range(5))
def test_no_implicit_dtype_conversion(index):
    args = metadata()
    args[index].dtype = mx.float16
    assert not hc.eligible(*args, **KW)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_coefficient_storage_and_hydrated_dtype(dtype):
    args = metadata()
    args[2].dtype = args[3].dtype = dtype
    assert hc.eligible(*args, **KW)
    args[3].dtype = mx.bfloat16 if dtype == mx.float32 else mx.float32
    assert not hc.eligible(*args, **KW)


@pytest.mark.parametrize("key,value", [
    ("rms_eps", float("nan")), ("norm_eps", float("inf")),
    ("sink_eps", -1), ("iterations", 19),
])
def test_scalar_guards(key, value):
    kw = dict(KW, **{key: value})
    assert not hc.eligible(*metadata(), **kw)


def test_default_on_explicit_off_and_scope_guards(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("declined path constructed a kernel")
    monkeypatch.setattr(hc, "_projection_kernel", forbidden)
    assert hc.glm5_mhc_norm_decode(*metadata(), **KW) is None
    monkeypatch.setattr(hc, "_CALLS", 0)
    monkeypatch.delenv("VMLX_GLM5_MHC_WEIGHTED_RMS", raising=False)
    assert hc.glm5_mhc_norm_status() == {"requested": True, "graph_calls": 0}
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: False)
    assert hc.try_glm5_hc_norm(None, None, None) is None
    monkeypatch.setenv("VMLX_GLM5_MHC_WEIGHTED_RMS", "0")
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: True)
    assert hc.try_glm5_hc_norm(None, None, None) is None
    assert hc.glm5_mhc_norm_status() == {"requested": False, "graph_calls": 0}


def test_admitted_failure_is_not_replayed(monkeypatch):
    failure = RuntimeError("kernel construction failed")
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise failure
    monkeypatch.setattr(hc, "_projection_kernel", lambda: fail)
    monkeypatch.setattr(hc, "_scalar", lambda value: value)
    with pytest.raises(RuntimeError) as caught:
        hc.glm5_mhc_norm_decode(*metadata(), **KW, enabled=True)
    assert caught.value is failure and len(calls) == 1


def test_production_does_not_declare_dead_collapsed_buffer(monkeypatch):
    hc._kernel.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: kw)
    try:
        production = hc._kernel(False)
        capture = hc._kernel(True)
        assert production["output_names"] == ["post", "comb", "normalized"]
        assert capture["output_names"] == ["post", "comb", "collapsed", "normalized"]
        assert "collapsed[collapsed_base + dim]" not in production["source"]
        assert "collapsed[collapsed_base + dim]" in capture["source"]
        assert "T rounded = T(value);" in production["source"]
        assert production["name"] != capture["name"]
    finally:
        hc._kernel.cache_clear()


def test_health_exposes_graph_construction_not_gpu_completion(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_read_bundle_json", lambda *args: {"model_type": "glm5_next"})
    monkeypatch.setattr(server, "_loaded_acceleration_attestation", lambda: None)
    monkeypatch.setattr(hc, "_CALLS", 90)
    monkeypatch.setenv("VMLX_GLM5_MHC_WEIGHTED_RMS", "1")
    contract = server._family_acceleration_contract(None)
    feature = next(row for row in contract["features"] if row["id"] == "mhc_transform")
    assert feature["runtime"]["compound_weighted_rms"] == {
        "graph_calls": 90, "requested": True,
    }


def assert_words_equal(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape
    word = mx.uint16 if got.dtype == mx.bfloat16 else mx.uint32
    np.testing.assert_array_equal(np.asarray(got.view(word)), np.asarray(want.view(word)))


@pytest.mark.parametrize("coefficient_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("pattern", ["random", "zero", "large", "cancellation"])
def test_component_all_outputs_word_exact(pattern, coefficient_dtype):
    if not mx.metal.is_available() or not hc._compatible_runtime():
        pytest.skip("requires MLX 0.32.2 or 0.32.3 / M5 Max; no source-host compute")
    rng = np.random.default_rng(53317)
    x = rng.normal(size=(1, 1, 4, 4096)).astype(np.float32)
    if pattern == "zero":
        x.fill(0)
    elif pattern == "large":
        x *= 256
    elif pattern == "cancellation":
        x[..., 1, :] = -x[..., 0, :]
        x[..., 3, :] = -x[..., 2, :]
    streams = mx.array(x, dtype=mx.bfloat16)
    fn = mx.array(rng.normal(0, 0.005, size=(24, 16384)), dtype=mx.bfloat16)
    dtype = getattr(mx, coefficient_dtype)
    base = mx.array(rng.normal(size=24), dtype=dtype)
    scale = mx.array([0.5, 1, 0.125], dtype=dtype)
    weight = mx.array(rng.uniform(-2, 2, size=4096), dtype=mx.bfloat16)
    post, comb, collapsed = glm5_mhc_decode(
        streams, fn, base, scale, rms_eps=KW["rms_eps"],
        sink_eps=KW["sink_eps"], iterations=20, enabled=True)
    expected = (post, comb, collapsed, mx.fast.rms_norm(collapsed, weight, KW["norm_eps"]))
    actual = hc.glm5_mhc_norm_decode(streams, fn, base, scale, weight, **KW,
                                    enabled=True, capture=True)
    production = hc.glm5_mhc_norm_decode(streams, fn, base, scale, weight, **KW, enabled=True)
    assert actual is not None and len(actual) == 4
    assert production is not None and len(production) == 3
    mx.eval(*expected, *actual, *production)
    for got, want in zip(actual, expected):
        assert_words_equal(got, want)
    for got, want in zip(production, (expected[0], expected[1], expected[3])):
        assert_words_equal(got, want)


@pytest.mark.parametrize("case", ["runtime", "prefill", "batch", "parent_disabled"])
def test_default_on_keeps_unsupported_paths_stock(monkeypatch, case):
    streams, fn, base, scale, weight = metadata()
    connection = SimpleNamespace(_fused_decode=case != "parent_disabled",
        hc_fn=fn, hc_base=base, hc_scale=scale, rms_eps=KW["rms_eps"],
        eps=KW["sink_eps"], iters=KW["iterations"])
    norm = SimpleNamespace(weight=weight, eps=KW["norm_eps"])
    if case == "prefill": streams.shape = (1, 2, 4, 4096)
    elif case == "batch": streams.shape = (2, 1, 4, 4096)
    monkeypatch.delenv("VMLX_GLM5_MHC_WEIGHTED_RMS", raising=False)
    monkeypatch.setattr(hc, "_CALLS", 0)
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(hc, "_compatible_runtime", lambda: case != "runtime")
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported default path constructed a kernel")
    monkeypatch.setattr(hc, "_projection_kernel", forbidden)
    assert hc.try_glm5_hc_norm(streams, connection, norm) is None
    assert hc.glm5_mhc_norm_status() == {"requested": True, "graph_calls": 0}


@pytest.mark.parametrize("setting,requested", [(None, True), ("0", False), ("1", True)])
def test_default_policy_selects_dispatch_without_claiming_gpu_execution(monkeypatch, setting, requested):
    streams, fn, base, scale, weight = metadata()
    connection = SimpleNamespace(_fused_decode=True, hc_fn=fn, hc_base=base,
        hc_scale=scale, rms_eps=KW["rms_eps"], eps=KW["sink_eps"], iters=20)
    norm = SimpleNamespace(weight=weight, eps=KW["norm_eps"])
    monkeypatch.delenv("VMLX_GLM5_MHC_WEIGHTED_RMS", raising=False)
    if setting is not None:
        monkeypatch.setenv("VMLX_GLM5_MHC_WEIGHTED_RMS", setting)
    monkeypatch.setattr(hc, "_CALLS", 0)
    monkeypatch.setattr(hc, "affine_moe_ar_scope_active", lambda: True)
    monkeypatch.setattr(hc, "_compatible_runtime", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    outputs = (object(), object(), object())
    calls = []
    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return outputs
    monkeypatch.setattr(hc, "glm5_mhc_norm_decode", dispatch)
    actual = hc.try_glm5_hc_norm(streams, connection, norm)
    assert actual is (outputs if requested else None)
    assert len(calls) == int(requested)
    assert hc.glm5_mhc_norm_status() == {
        "requested": requested, "graph_calls": int(requested)}
    if calls:
        assert calls[0][1]["enabled"] is True


@pytest.mark.parametrize("version,device,expected", [
    ("0.32.2", "Apple M5 Max", True),
    ("0.32.3", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M5 Max", True),
    ("0.32.5.dev20261101+360078a8", "Apple M5 Ultra", False),
    ("0.32.1", "Apple M5 Max", False),
    ("0.32.4", "Apple M5 Max", False),
    ("0.32.3", "Apple M4 Max", False),
])
def test_qualified_runtime_versions_only(monkeypatch, version, device, expected):
    monkeypatch.setattr(hc.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": device})
    hc._compatible_runtime.cache_clear()
    try:
        assert hc._compatible_runtime() is expected
    finally:
        hc._compatible_runtime.cache_clear()

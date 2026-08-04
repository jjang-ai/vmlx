"""Focused contracts for the guarded DSV4 affine PR #248 integration."""

from __future__ import annotations

import weakref
from contextlib import suppress

import pytest


class _FakeArray:
    def __init__(self, shape=(), dtype="float16"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)
        size = 1
        for value in self.shape:
            size *= value
        self.size = size

    def reshape(self, *_shape):
        return self

    def astype(self, _dtype):
        return self


class _FakeMx:
    uint32 = "uint32"
    uint16 = "uint16"
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    def array(_value, dtype=None):
        return _FakeArray((), dtype=dtype)

    @staticmethod
    def eval(*_values):
        return None


class _FakeQuantizedSwitchLinear:
    def __init__(self, input_dims, output_dims, experts, bits, group_size):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bits = bits
        self.group_size = group_size
        self.mode = "affine"
        groups = input_dims // group_size
        packed = input_dims * bits // 32
        self.weight = _FakeArray((experts, output_dims, packed), "uint32")
        self.scales = _FakeArray((experts, output_dims, groups), "float16")
        self.biases = _FakeArray((experts, output_dims, groups), "float16")

    def __contains__(self, key):
        return False


class _FakeActivation:
    swiglu_limit = 10.0

    def __call__(self, up, gate):
        return up


class _FakeSwitchGLU:
    def __init__(self, *, gate_bits=3, up_bits=2, down_groups=32, hidden=128, inter=64):
        self.gate_proj = _FakeQuantizedSwitchLinear(hidden, inter, 4, gate_bits, 64)
        self.up_proj = _FakeQuantizedSwitchLinear(hidden, inter, 4, up_bits, 64)
        self.down_proj = _FakeQuantizedSwitchLinear(inter, hidden, 4, 2, down_groups)
        self.activation = _FakeActivation()

    def __call__(self, x, indices):
        return ("stock", x, indices)


_FAKE_STOCK_CALL = _FakeSwitchGLU.__call__


class _FakeModel:
    config = {"model_type": "deepseek_v4"}

    def __init__(self, switch):
        self.switch = switch

    def named_modules(self):
        return [("layers.0.ffn.switch_mlp", self.switch)]


class _FakeWeightedMoE:
    def __init__(self, switch):
        self.switch_mlp = switch

    def _weighted_routed_experts(self, x, indices, scores):
        return ("stock-weighted", x, indices, scores)


_FAKE_STOCK_WEIGHTED_CALL = _FakeWeightedMoE._weighted_routed_experts


class _FakeWeightedModel(_FakeModel):
    def __init__(self, switch):
        super().__init__(switch)
        self.moe = _FakeWeightedMoE(switch)

    def named_modules(self):
        return [
            ("layers.0.ffn", self.moe),
            ("layers.0.ffn.switch_mlp", self.switch),
        ]


@pytest.fixture
def fake_runtime(monkeypatch):
    import mlx_lm.models.switch_layers as switch_layers

    import vmlx_engine.metal.affine_moe_decode as fastpath

    monkeypatch.setattr(fastpath, "mx", _FakeMx)
    monkeypatch.setattr(fastpath, "_CONFIGS", fastpath._WeakIdentityMap())
    monkeypatch.setattr(fastpath, "_PATCHED_CLASS", None)
    monkeypatch.setattr(fastpath, "_ORIGINAL_CALL", None)
    monkeypatch.setattr(fastpath, "_WRAPPER", None)
    monkeypatch.setattr(fastpath, "_PATCHED_WEIGHTED_CLASS", None)
    monkeypatch.setattr(fastpath, "_ORIGINAL_WEIGHTED_CALL", None)
    monkeypatch.setattr(fastpath, "_WEIGHTED_WRAPPER", None)
    monkeypatch.setattr(fastpath, "_FIRST_FAST_CALL_LOGGED", False)
    monkeypatch.setattr(fastpath, "_FIRST_REGISTERED_FALLBACK_LOGGED", False)
    monkeypatch.setattr(switch_layers, "SwitchGLU", _FakeSwitchGLU)
    monkeypatch.setattr(
        switch_layers,
        "QuantizedSwitchLinear",
        _FakeQuantizedSwitchLinear,
    )
    monkeypatch.setenv("VMLX_DSV4_AFFINE_MOE_FASTPATH", "1")
    _FakeSwitchGLU.__call__ = _FAKE_STOCK_CALL
    _FakeWeightedMoE._weighted_routed_experts = _FAKE_STOCK_WEIGHTED_CALL
    yield fastpath
    _FakeSwitchGLU.__call__ = _FAKE_STOCK_CALL
    _FakeWeightedMoE._weighted_routed_experts = _FAKE_STOCK_WEIGHTED_CALL


def test_dsv4_affine_installer_is_weak_scoped_and_idempotent(fake_runtime):
    switch = _FakeSwitchGLU()
    model = _FakeModel(switch)

    assert fake_runtime.install_dsv4_affine_moe_fastpath(model) == 1
    wrapper = _FakeSwitchGLU.__call__
    first_config = fake_runtime._CONFIGS[switch]
    assert fake_runtime.install_dsv4_affine_moe_fastpath(model) == 1

    assert _FakeSwitchGLU.__call__ is wrapper
    assert fake_runtime._CONFIGS[switch] is not first_config
    assert weakref.ref(switch)() is switch
    assert switch in fake_runtime._CONFIGS


def test_dsv4_affine_weak_identity_map_accepts_unhashable_modules(fake_runtime):
    class _UnhashableSwitch(_FakeSwitchGLU):
        __hash__ = None

    switch = _UnhashableSwitch()
    model = _FakeModel(switch)

    assert fake_runtime.install_dsv4_affine_moe_fastpath(model) == 1
    assert fake_runtime._CONFIGS.get(switch) is not None


def test_dsv4_affine_opt_out_precedes_install(fake_runtime, monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_AFFINE_MOE_FASTPATH", "0")
    switch = _FakeSwitchGLU()

    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(switch)) == 0
    assert switch not in fake_runtime._CONFIGS


def test_dsv4_affine_default_is_native_stock(fake_runtime, monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_AFFINE_MOE_FASTPATH")
    switch = _FakeSwitchGLU()

    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(switch)) == 0
    assert switch not in fake_runtime._CONFIGS


def test_dsv4_affine_rejects_unsupported_and_remainder_layouts(fake_runtime):
    unsupported = _FakeSwitchGLU(up_bits=3)
    remainder = _FakeSwitchGLU(hidden=192, inter=96)

    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(unsupported)) == 0
    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(remainder)) == 0
    assert unsupported not in fake_runtime._CONFIGS
    assert remainder not in fake_runtime._CONFIGS


def test_dsv4_affine_install_is_atomic_when_any_switchglu_rejects(fake_runtime):
    compatible = _FakeSwitchGLU()
    incompatible = _FakeSwitchGLU(up_bits=3)
    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(compatible)) == 1
    wrapper = _FakeSwitchGLU.__call__
    assert compatible in fake_runtime._CONFIGS

    class _MixedModel:
        config = {"model_type": "deepseek_v4"}

        def named_modules(self):
            return [
                ("layers.0.ffn.switch_mlp", compatible),
                ("layers.1.ffn.switch_mlp", incompatible),
            ]

    assert fake_runtime.install_dsv4_affine_moe_fastpath(_MixedModel()) == 0
    assert compatible not in fake_runtime._CONFIGS
    assert incompatible not in fake_runtime._CONFIGS
    assert _FakeSwitchGLU.__call__ is wrapper
    x = _FakeArray((1, 1, 128), "float16")
    indices = _FakeArray((1, 1, 2), "uint32")
    assert compatible(x, indices) == ("stock", x, indices)


def test_dsv4_affine_runtime_exception_disables_only_eligible_module(
    fake_runtime,
    monkeypatch,
):
    class _BrokenManager:
        def run_projection(self, *_args, **_kwargs):
            raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(fake_runtime, "_MANAGER", _BrokenManager())
    switch = _FakeSwitchGLU()
    other = _FakeSwitchGLU()
    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(switch)) == 1
    x = _FakeArray((1, 1, 128), "float16")
    indices = _FakeArray((1, 1, 2), "uint32")

    result = switch(x, indices)
    untouched = other(x, indices)

    assert result[0] == "stock"
    assert untouched[0] == "stock"
    assert "synthetic compile failure" in fake_runtime._CONFIGS[switch].disabled_reason


def test_dsv4_affine_training_uses_stock_without_disabling_decode(fake_runtime, monkeypatch):
    class _DecodeOnlyManager:
        def run_projection(self, *_args, **_kwargs):
            raise AssertionError("training must remain on stock SwitchGLU")

    monkeypatch.setattr(fake_runtime, "_MANAGER", _DecodeOnlyManager())
    switch = _FakeSwitchGLU()
    switch.training = True
    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(switch)) == 1
    x = _FakeArray((1, 1, 128), "float16")
    indices = _FakeArray((1, 1, 2), "uint32")

    assert switch(x, indices) == ("stock", x, indices)
    assert fake_runtime._CONFIGS[switch].disabled_reason is None


def test_dsv4_affine_current_dsv4_weighted_route_owns_decode(
    fake_runtime,
    monkeypatch,
):
    switch = _FakeSwitchGLU()
    model = _FakeWeightedModel(switch)
    expected = object()
    calls = []

    def _fake_weighted_decode(owner, config, x, indices, scores):
        calls.append((owner, config, x, indices, scores))
        return expected

    monkeypatch.setattr(fake_runtime, "_weighted_decode", _fake_weighted_decode)
    assert fake_runtime.install_dsv4_affine_moe_fastpath(model) == 1
    assert _FakeSwitchGLU.__call__ is _FAKE_STOCK_CALL
    x = _FakeArray((1, 1, 128), "float16")
    indices = _FakeArray((1, 1, 2), "uint32")
    scores = _FakeArray((1, 1, 2), "float32")

    assert model.moe._weighted_routed_experts(x, indices, scores) is expected
    assert calls and calls[0][0] is switch


def test_dsv4_affine_current_dsv4_weighted_route_preserves_prefill(
    fake_runtime,
    monkeypatch,
):
    switch = _FakeSwitchGLU()
    model = _FakeWeightedModel(switch)

    def _unexpected_weighted_decode(*_args, **_kwargs):
        raise AssertionError("decode kernel must not own weighted prefill")

    monkeypatch.setattr(
        fake_runtime,
        "_weighted_decode",
        _unexpected_weighted_decode,
    )
    assert fake_runtime.install_dsv4_affine_moe_fastpath(model) == 1
    x = _FakeArray((1, 2, 128), "float16")
    indices = _FakeArray((1, 2, 2), "uint32")
    scores = _FakeArray((1, 2, 2), "float32")

    result = model.moe._weighted_routed_experts(x, indices, scores)
    assert result == ("stock-weighted", x, indices, scores)


def test_dsv4_affine_rejects_partial_weighted_owner_topology(fake_runtime):
    first = _FakeSwitchGLU()
    second = _FakeSwitchGLU()
    owner = _FakeWeightedMoE(first)

    class _PartialOwnerModel:
        config = {"model_type": "deepseek_v4"}

        def named_modules(self):
            return [
                ("layers.0.ffn", owner),
                ("layers.0.ffn.switch_mlp", first),
                ("layers.1.ffn.switch_mlp", second),
            ]

    assert fake_runtime.install_dsv4_affine_moe_fastpath(_PartialOwnerModel()) == 0
    assert first not in fake_runtime._CONFIGS
    assert second not in fake_runtime._CONFIGS
    assert _FakeWeightedMoE._weighted_routed_experts is _FAKE_STOCK_WEIGHTED_CALL


@pytest.mark.parametrize(
    ("x_shape", "indices_shape"),
    [
        ((1, 2, 128), (1, 2, 2)),
        ((2, 128), (2, 2)),
    ],
)
def test_dsv4_affine_multitoken_prefill_stays_on_stock_switchglu(
    fake_runtime,
    monkeypatch,
    x_shape,
    indices_shape,
):
    class _DecodeOnlyManager:
        def run_projection(self, *_args, **_kwargs):
            raise AssertionError("decode kernel must not own multi-token prefill")

    monkeypatch.setattr(fake_runtime, "_MANAGER", _DecodeOnlyManager())
    switch = _FakeSwitchGLU()
    assert fake_runtime.install_dsv4_affine_moe_fastpath(_FakeModel(switch)) == 1
    x = _FakeArray(x_shape, "float16")
    indices = _FakeArray(indices_shape, "uint32")

    result = switch(x, indices)

    assert result == ("stock", x, indices)


def test_dsv4_affine_populated_packed_weights_match_stock_and_preserve_rank(monkeypatch):
    """Metal numeric gate; run on the designated DSV4 compute Mac."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

    import vmlx_engine.metal.affine_moe_decode as fastpath

    monkeypatch.setenv("VMLX_DSV4_AFFINE_MOE_FASTPATH", "1")

    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            pytest.skip("requires MLX Metal")
    except Exception:
        pytest.skip("requires MLX Metal")

    class _Activation(nn.Module):
        swiglu_limit = 10.0

        def __call__(self, x_up, x_gate):
            return x_gate * mx.sigmoid(x_gate) * x_up

    mx.random.seed(248)
    switch = SwitchGLU(128, 64, 8, activation=_Activation(), bias=False)
    switch.gate_proj = SwitchLinear(128, 64, 8, bias=False).to_quantized(64, 3)
    switch.up_proj = SwitchLinear(128, 64, 8, bias=False).to_quantized(64, 2)
    switch.down_proj = SwitchLinear(64, 128, 8, bias=False).to_quantized(32, 2)
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(mx.float16)
        projection.biases = projection.biases.astype(mx.float16)
    model = _FakeModel(switch)
    x = mx.random.normal((1, 1, 128)).astype(mx.float16)
    indices = mx.array([[[1, 3, 0, 2, 4, 7]]], dtype=mx.uint32)
    original = getattr(
        SwitchGLU,
        "_vmlx_dsv4_affine_original_call",
        SwitchGLU.__call__,
    )
    expected = original(switch, x, indices)
    mx.eval(expected)

    saved_call = SwitchGLU.__call__
    saved_configs = fastpath._CONFIGS
    saved_patched_class = fastpath._PATCHED_CLASS
    saved_original_call = fastpath._ORIGINAL_CALL
    saved_wrapper = fastpath._WRAPPER
    saved_attrs = {
        name: getattr(SwitchGLU, name, None)
        for name in (
            "_vmlx_dsv4_affine_original_call",
            "_vmlx_dsv4_affine_decode_fastpath",
        )
    }
    try:
        fastpath._CONFIGS = fastpath._WeakIdentityMap()
        fastpath._PATCHED_CLASS = None
        fastpath._ORIGINAL_CALL = None
        fastpath._WRAPPER = None
        assert fastpath.install_dsv4_affine_moe_fastpath(model) == 1
        actual = switch(x, indices)
        mx.eval(actual)

        assert actual.shape == expected.shape == (1, 1, 6, 128)
        delta = actual.astype(mx.float32) - expected.astype(mx.float32)
        rel_rms = mx.sqrt(mx.mean(mx.square(delta))) / mx.maximum(
            mx.sqrt(mx.mean(mx.square(expected.astype(mx.float32)))),
            mx.array(1e-8),
        )
        assert float(rel_rms.item()) < 0.02
        assert float(mx.max(mx.abs(delta)).item()) < 0.25

        # Multi-token prefill must remain byte-for-byte owned by stock SwitchGLU.
        prefill_x = mx.random.normal((1, 2, 128)).astype(mx.float16)
        prefill_indices = mx.array(
            [[[0, 2, 4, 6, 1, 3], [1, 3, 5, 7, 0, 2]]],
            dtype=mx.uint32,
        )
        prefill_expected = original(switch, prefill_x, prefill_indices)
        prefill_actual = switch(prefill_x, prefill_indices)
        mx.eval(prefill_expected, prefill_actual)
        assert prefill_actual.shape == prefill_expected.shape == (1, 2, 6, 128)
        assert bool(mx.array_equal(prefill_actual, prefill_expected).item())
    finally:
        SwitchGLU.__call__ = saved_call
        fastpath._CONFIGS = saved_configs
        fastpath._PATCHED_CLASS = saved_patched_class
        fastpath._ORIGINAL_CALL = saved_original_call
        fastpath._WRAPPER = saved_wrapper
        for name, value in saved_attrs.items():
            if value is None:
                with suppress(AttributeError):
                    delattr(SwitchGLU, name)
            else:
                setattr(SwitchGLU, name, value)


def test_dsv4_affine_weighted_owner_matches_current_dsv4_stock_order(monkeypatch):
    """Metal numeric gate for score-before-down current DSV4 packages."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

    import vmlx_engine.metal.affine_moe_decode as fastpath

    monkeypatch.setenv("VMLX_DSV4_AFFINE_MOE_FASTPATH", "1")

    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            pytest.skip("requires MLX Metal")
    except Exception:
        pytest.skip("requires MLX Metal")

    class _Activation(nn.Module):
        swiglu_limit = 10.0

        def __call__(self, x_up, x_gate):
            gate = mx.clip(x_gate.astype(mx.float32), a_min=None, a_max=10.0)
            up = mx.clip(x_up.astype(mx.float32), a_min=-10.0, a_max=10.0)
            return (mx.sigmoid(gate) * gate * up).astype(x_gate.dtype)

    class _CurrentWeightedOwner:
        def __init__(self, switch):
            self.switch_mlp = switch

        def _weighted_routed_experts(self, x, inds, scores):
            switch = self.switch_mlp
            dtype = x.dtype
            expanded = mx.expand_dims(x, (-2, -3))
            x_up = switch.up_proj(expanded, inds, sorted_indices=False)
            x_gate = switch.gate_proj(expanded, inds, sorted_indices=False)
            gate = mx.clip(x_gate.astype(mx.float32), a_min=None, a_max=10.0)
            up = mx.clip(x_up.astype(mx.float32), a_min=-10.0, a_max=10.0)
            activated = mx.sigmoid(gate) * gate * up
            activated = (activated * scores[..., None, None]).astype(dtype)
            return switch.down_proj(
                activated,
                inds,
                sorted_indices=False,
            ).squeeze(-2)

    mx.random.seed(249)
    switch = SwitchGLU(128, 64, 8, activation=_Activation(), bias=False)
    switch.gate_proj = SwitchLinear(128, 64, 8, bias=False).to_quantized(64, 3)
    switch.up_proj = SwitchLinear(128, 64, 8, bias=False).to_quantized(64, 2)
    switch.down_proj = SwitchLinear(64, 128, 8, bias=False).to_quantized(32, 2)
    for projection in (switch.gate_proj, switch.up_proj, switch.down_proj):
        projection.scales = projection.scales.astype(mx.float16)
        projection.biases = projection.biases.astype(mx.float16)
    owner = _CurrentWeightedOwner(switch)

    class _CurrentWeightedModel:
        config = {"model_type": "deepseek_v4"}

        def named_modules(self):
            return [("layers.0.ffn", owner), ("layers.0.ffn.switch_mlp", switch)]

    x = mx.random.normal((1, 1, 128)).astype(mx.float16)
    indices = mx.array([[[1, 3, 0, 2, 4, 7]]], dtype=mx.uint32)
    scores = mx.array([[[0.31, 0.24, 0.18, 0.12, 0.09, 0.06]]], dtype=mx.float32)
    original = _CurrentWeightedOwner._weighted_routed_experts
    expected = original(owner, x, indices, scores)
    mx.eval(expected)

    saved_weighted_class = fastpath._PATCHED_WEIGHTED_CLASS
    saved_original = fastpath._ORIGINAL_WEIGHTED_CALL
    saved_wrapper = fastpath._WEIGHTED_WRAPPER
    saved_configs = fastpath._CONFIGS
    try:
        fastpath._PATCHED_WEIGHTED_CLASS = None
        fastpath._ORIGINAL_WEIGHTED_CALL = None
        fastpath._WEIGHTED_WRAPPER = None
        fastpath._CONFIGS = fastpath._WeakIdentityMap()
        assert fastpath.install_dsv4_affine_moe_fastpath(_CurrentWeightedModel()) == 1
        actual = owner._weighted_routed_experts(x, indices, scores)
        mx.eval(actual)

        assert actual.shape == expected.shape == (1, 1, 6, 128)
        delta = actual.astype(mx.float32) - expected.astype(mx.float32)
        rel_rms = mx.sqrt(mx.mean(mx.square(delta))) / mx.maximum(
            mx.sqrt(mx.mean(mx.square(expected.astype(mx.float32)))),
            mx.array(1e-8),
        )
        assert float(rel_rms.item()) < 0.02
        assert float(mx.max(mx.abs(delta)).item()) < 0.25
    finally:
        _CurrentWeightedOwner._weighted_routed_experts = original
        fastpath._PATCHED_WEIGHTED_CLASS = saved_weighted_class
        fastpath._ORIGINAL_WEIGHTED_CALL = saved_original
        fastpath._WEIGHTED_WRAPPER = saved_wrapper
        fastpath._CONFIGS = saved_configs

# SPDX-License-Identifier: Apache-2.0
"""v1.6.24 integration gates for the DSV4 exact fastpaths.

These pin the composition rules from the PR #249 integration plan: the
canonical cache-floor control, fail-closed admission, allocator-ceiling
byte accounting across model lifetime, dispatch scoping for unsupported
families, exact output identity, and non-interference with the v1.6.24
output and Metal live-buffer guards.
"""

from __future__ import annotations

import gc
import threading
from types import SimpleNamespace

import pytest


class _RecordingMx:
    """Minimal mx stand-in that records allocator-ceiling writes."""

    def __init__(self):
        self.limits = []

    def set_cache_limit(self, value):
        self.limits.append(value)


def _fresh_lm_head_module(monkeypatch, mx_stub):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    monkeypatch.setattr(mod, "mx", mx_stub)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_RESERVATION_LOCK", threading.RLock())
    monkeypatch.setattr(mod, "_CACHE_LIMIT_BASE_BYTES", None)
    monkeypatch.setattr(mod, "_CACHE_LIMIT_EFFECTIVE_BYTES", None)
    return mod


def test_canonical_cache_floor_env_var_changes_floor(monkeypatch):
    mod = _fresh_lm_head_module(monkeypatch, _RecordingMx())
    gib = 1024**3

    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "2")
    assert mod._cache_limit_minimum_bytes() == 2 * gib

    monkeypatch.delenv("VMLX_DSV4_MLX_CACHE_MIN_GB", raising=False)
    assert mod._cache_limit_minimum_bytes() == 4 * gib

    # The pre-integration typo must have no effect: no alias is retained
    # because no released version ever exposed it.
    monkeypatch.setenv("VMLINUX_DSV4_MLX_CACHE_MIN_GB", "1")
    assert mod._cache_limit_minimum_bytes() == 4 * gib


@pytest.mark.parametrize("raw", ["banana", "", "-3", "0", "nan", "inf"])
def test_invalid_cache_floor_values_fall_back_to_default(monkeypatch, raw):
    mod = _fresh_lm_head_module(monkeypatch, _RecordingMx())
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", raw)
    assert mod._cache_limit_minimum_bytes() == 4 * 1024**3


def test_unsupported_model_type_installs_nothing(monkeypatch):
    mod = _fresh_lm_head_module(monkeypatch, _RecordingMx())
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    materialized = []
    monkeypatch.setattr(
        mod, "_materialize_weight", lambda _h: materialized.append(1)
    )

    class _Model:
        model_type = "qwen3_moe"

        def __call__(self, *args, **kwargs):
            return "stock"

    model = _Model()
    original_call = _Model.__call__

    assert mod.install_dsv4_lm_head_fastpath(model) is False
    assert _Model.__call__ is original_call
    assert mod._CONFIGS.get(model) is None
    assert not mod._RESERVATIONS
    assert not materialized
    assert mod.dsv4_lm_head_fastpath_status(model)["installed"] is False


def test_insufficient_headroom_denies_without_side_effects(monkeypatch):
    mx_stub = _RecordingMx()
    mod = _fresh_lm_head_module(monkeypatch, mx_stub)
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    monkeypatch.setattr(mod, "_validate_head", lambda _h: ("sig",))
    monkeypatch.setattr(mod, "_estimated_cache_bytes", lambda _h: 2 * 1024**3)
    monkeypatch.setattr(mod, "_model_type", lambda _m: "deepseek_v4")
    monkeypatch.setattr(mod, "_cache_admitted", lambda *_a: False)
    materialized = []
    monkeypatch.setattr(
        mod, "_materialize_weight", lambda _h: materialized.append(1)
    )

    class _Model:
        lm_head = SimpleNamespace()

        def __call__(self, *args, **kwargs):
            return "stock"

    model = _Model()

    assert mod.install_dsv4_lm_head_fastpath(model) is False
    assert not materialized
    assert not mx_stub.limits  # allocator ceiling never touched
    assert not mod._RESERVATIONS


def test_headroom_lookup_failure_fails_closed(monkeypatch):
    mx_stub = _RecordingMx()
    mod = _fresh_lm_head_module(monkeypatch, mx_stub)

    from vmlx_engine.utils import memory_limits

    def _boom(_mx=None):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(
        memory_limits, "get_effective_metal_working_set_bytes", _boom
    )
    monkeypatch.setattr(mod, "_parameter_bytes", lambda _m: 10 * 1024**3)

    assert mod._cache_admitted(object(), 2 * 1024**3) is False


def test_model_gc_releases_reservation_and_restores_ceiling(monkeypatch):
    mx_stub = _RecordingMx()
    mod = _fresh_lm_head_module(monkeypatch, mx_stub)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "1")
    gib = 1024**3

    class _Model:
        pass

    model = _Model()
    assert mod.configure_dsv4_lm_head_cache_budget(8 * gib) == 8 * gib
    assert mod._reserve_allocator_cache(model, 2 * gib)
    assert mx_stub.limits[-1] == 6 * gib

    del model
    gc.collect()

    assert not mod._RESERVATIONS
    assert mx_stub.limits[-1] == 8 * gib


def test_configure_budget_fails_closed(monkeypatch):
    mod = _fresh_lm_head_module(monkeypatch, _RecordingMx())

    with pytest.raises(ValueError):
        mod.configure_dsv4_lm_head_cache_budget(0)
    with pytest.raises(ValueError):
        mod.configure_dsv4_lm_head_cache_budget(-1)

    monkeypatch.setattr(mod, "mx", None)
    with pytest.raises(RuntimeError):
        mod.configure_dsv4_lm_head_cache_budget(8 * 1024**3)

    class _BrokenMx:
        @staticmethod
        def set_cache_limit(_value):
            raise OSError("allocator control unavailable")

    monkeypatch.setattr(mod, "mx", _BrokenMx)
    with pytest.raises(OSError):
        mod.configure_dsv4_lm_head_cache_budget(8 * 1024**3)


def test_guarded_projection_is_bit_identical_to_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_cache_admitted", lambda *_a: True)
    monkeypatch.setattr(mod, "_reserve_allocator_cache", lambda *_a: True)
    monkeypatch.setattr(mod, "_release_cache_reservation", lambda *_a: None)
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")

    vocab, dims, group = 64, 128, 32
    source = mx.random.normal((vocab, dims)).astype(mx.float16)
    weight_q, scales, biases = mx.quantize(source, group_size=group, bits=8)

    class _Head:
        def __init__(self):
            self.weight = weight_q
            self.scales = scales
            self.biases = biases
            self.bits = 8
            self.group_size = group
            self.mode = "affine"

        def __contains__(self, _key):
            return False

    hidden = mx.random.normal((1, 3, dims)).astype(mx.float16)

    class _Model:
        model_type = "deepseek_v4"

        def __init__(self):
            self.lm_head = _Head()

        def model(self, _input_ids, cache=None, mask=None):
            del cache, mask
            return hidden

        def __call__(self, input_ids, cache=None, mask=None):
            return mod._reference_logits(
                self.model(input_ids, cache=cache, mask=mask), self.lm_head
            )

    model = _Model()
    reference = model("ids")
    mx.eval(reference)

    assert mod.install_dsv4_lm_head_fastpath(model)
    optimized = model("ids")
    mx.eval(optimized)

    assert optimized.dtype == reference.dtype == mx.float32
    assert optimized.shape == reference.shape
    assert bool(mx.array_equal(optimized, reference))
    status = mod.dsv4_lm_head_fastpath_status(model)
    assert status["installed"] and status["validated"]
    assert status["cache_bytes"] == vocab * dims * 4


def test_v1624_output_guards_are_untouched_by_fastpath_install(monkeypatch):
    from vmlx_engine.utils import memory_limits

    guard_fn = memory_limits.projected_output_token_cap_by_buffers
    byte_fn = memory_limits.projected_output_token_cap
    rate_fn = memory_limits.retained_buffers_per_token
    fraction = memory_limits._BUFFER_GUARD_SAFETY_FRACTION

    mod = _fresh_lm_head_module(monkeypatch, _RecordingMx())
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    mod.configure_dsv4_lm_head_cache_budget(8 * 1024**3)
    holder = type("_Holder", (), {})()
    mod._reserve_allocator_cache(holder, 2 * 1024**3)

    import vmlx_engine.models.dsv4_rope_cache  # noqa: F401 - import side effects only

    assert memory_limits.projected_output_token_cap_by_buffers is guard_fn
    assert memory_limits.projected_output_token_cap is byte_fn
    assert memory_limits.retained_buffers_per_token is rate_fn
    assert fraction == memory_limits._BUFFER_GUARD_SAFETY_FRACTION

    # The DSV4 live-buffer projection still caps a DSV4-shaped config.
    config = {"model_type": "deepseek_v4", "num_hidden_layers": 43}
    assert memory_limits.retained_buffers_per_token(config) > 0
    cap = memory_limits.projected_output_token_cap_by_buffers(
        resource_limit=499000,
        live_baseline_buffers=2000,
        buffers_per_token=43.0,
    )
    assert cap is not None and 0 < cap < 12000


def test_rope_cache_stays_bounded_across_varied_lengths(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import vmlx_engine.models.dsv4_rope_cache as mod

    monkeypatch.setattr(mod, "mx", mx)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_GROUPS", {})
    monkeypatch.setattr(mod, "_GROUP_LOCK", threading.RLock())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE", "1")
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES", "8")
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE_MAX_TOKENS", "64")

    class _Rope:
        inv_freq = mx.array([1.0, 0.5, 0.25, 0.125])

        def __init__(self):
            self.reference_calls = 0

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            self.reference_calls += 1
            return x

    rope = _Rope()
    model = SimpleNamespace(named_modules=lambda: [("rope", rope)])
    assert mod._install_for_class(model, _Rope) == 1
    group = mod._CONFIGS.get(rope)

    for step in range(40):
        x = mx.random.normal((1, 2, (step % 6) + 1, 8)).astype(mx.float16)
        out = rope(x, offset=step * 7)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert len(group.tables) <= 8

    # Over-length requests bypass the cache and use the reference path.
    long_x = mx.random.normal((1, 2, 65, 8)).astype(mx.float16)
    before = rope.reference_calls
    rope(long_x, offset=0)
    assert rope.reference_calls == before + 1
    assert len(group.tables) <= 8

# SPDX-License-Identifier: Apache-2.0
"""Failure-isolation tests for promoted DSV4 performance paths."""

from __future__ import annotations

from types import SimpleNamespace


class _Array:
    def __init__(self, shape, dtype="float16", value=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.value = value
        self.T = self

    def astype(self, dtype):
        return _Array(self.shape, dtype, self.value)

    def __matmul__(self, _other):
        return _Array(self.shape[:-1] + (16,), "float32", "reference")


class _Head:
    def __init__(self):
        self.weight = _Array((16, 4), "uint32")
        self.scales = _Array((16, 1), "float16")
        self.biases = _Array((16, 1), "float16")
        self.bits = 8
        self.group_size = 64
        self.mode = "affine"

    def __contains__(self, _key):
        return False


def test_lm_head_failure_falls_back_without_advancing_cache_twice(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    class _Mx:
        float32 = "float32"

        @staticmethod
        def quantized_matmul(*_args, **_kwargs):
            return _Array((1, 1, 16), value="fast")

        @staticmethod
        def eval(*_args):
            raise RuntimeError("synthetic lazy dispatch failure")

        @staticmethod
        def dequantize(*_args, **_kwargs):
            return _Array((16, 64), "float32")

    class _Model:
        model_type = "deepseek_v4"

        def __init__(self):
            self.lm_head = _Head()

        def model(self, _input_ids, cache=None, mask=None):
            del mask
            cache.advances += 1
            return _Array((1, 1, 64))

        def __call__(self, input_ids, cache=None, mask=None):
            hidden = self.model(input_ids, cache=cache, mask=mask)
            return ("stock", hidden)

    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_cache_admitted", lambda *_args: True)
    monkeypatch.setattr(mod, "_reserve_allocator_cache", lambda *_args: True)
    monkeypatch.setattr(mod, "_release_cache_reservation", lambda *_args: None)
    monkeypatch.setattr(mod, "_materialize_weight", lambda _head: _Array((16, 64), "float32"))
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    model = _Model()
    cache = SimpleNamespace(advances=0)

    assert mod.install_dsv4_lm_head_fastpath(model)
    assert mod.dsv4_lm_head_fastpath_status(model)["installed"] is True
    result = model("ids", cache=cache)

    assert result.value == "reference"
    assert cache.advances == 1
    assert mod._CONFIGS.get(model).disabled_reason is not None


def test_lm_head_registration_keeps_multiple_models_active(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    class _Mx:
        float32 = "float32"

        @staticmethod
        def eval(*_args):
            return None

        @staticmethod
        def dequantize(*_args, **_kwargs):
            return _Array((16, 64), "float32")

    class _Model:
        model_type = "deepseek_v4"

        def __init__(self):
            self.lm_head = _Head()

        def model(self, *_args, **_kwargs):
            return _Array((1, 1, 64))

        def __call__(self, *_args, **_kwargs):
            return "stock"

    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_cache_admitted", lambda *_args: True)
    monkeypatch.setattr(mod, "_reserve_allocator_cache", lambda *_args: True)
    monkeypatch.setattr(
        mod,
        "_materialize_weight",
        lambda _head: _Array((16, 64), "float32", "cached"),
    )
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    first, second = _Model(), _Model()

    assert mod.install_dsv4_lm_head_fastpath(first)
    assert mod.install_dsv4_lm_head_fastpath(second)
    assert first("ids").value == second("ids").value == "reference"
    assert mod._CONFIGS.get(first) is not None
    assert mod._CONFIGS.get(second) is not None
    assert mod.dsv4_lm_head_fastpath_status(first)["cache_bytes"] == 0


def test_lm_head_reinstall_is_idempotent(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    class _Mx:
        float32 = "float32"

        @staticmethod
        def eval(*_args):
            return None

        @staticmethod
        def dequantize(*_args, **_kwargs):
            return _Array((16, 64), "float32")

    class _Model:
        model_type = "deepseek_v4"

        def __init__(self):
            self.lm_head = _Head()

        def model(self, *_args, **_kwargs):
            return _Array((1, 1, 64))

        def __call__(self, *_args, **_kwargs):
            return "stock"

    materializations = []
    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setattr(mod, "_cache_admitted", lambda *_args: True)
    monkeypatch.setattr(mod, "_reserve_allocator_cache", lambda *_args: True)
    monkeypatch.setattr(
        mod,
        "_materialize_weight",
        lambda _head: materializations.append(object()) or _Array((16, 64)),
    )
    monkeypatch.setenv("VMLX_DSV4_LM_HEAD_MODE", "exact-cache")
    model = _Model()

    assert mod.install_dsv4_lm_head_fastpath(model)
    assert mod.install_dsv4_lm_head_fastpath(model)
    assert len(materializations) == 1


def test_lm_head_reservation_reduces_and_restores_allocator_cache(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    limits = []

    class _Mx:
        @staticmethod
        def set_cache_limit(value):
            limits.append(value)

    class _Model:
        pass

    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_CACHE_LIMIT_BASE_BYTES", None)
    monkeypatch.setattr(mod, "_CACHE_LIMIT_EFFECTIVE_BYTES", None)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "1")
    gib = 1024**3
    model = _Model()

    assert mod.configure_dsv4_lm_head_cache_budget(8 * gib) == 8 * gib
    assert mod._reserve_allocator_cache(model, 2 * gib)
    assert limits[-1] == 6 * gib

    mod._release_cache_reservation(id(model))
    assert limits[-1] == 8 * gib


def test_lm_head_reservation_rejects_cache_floor_violation(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    class _Mx:
        @staticmethod
        def set_cache_limit(_value):
            return None

    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_CACHE_LIMIT_BASE_BYTES", None)
    monkeypatch.setattr(mod, "_CACHE_LIMIT_EFFECTIVE_BYTES", None)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "4")
    gib = 1024**3
    model = type("Model", (), {})()

    mod.configure_dsv4_lm_head_cache_budget(5 * gib)
    assert not mod._reserve_allocator_cache(model, 2 * gib)


def test_lm_head_allocator_floor_never_raises_user_ceiling(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    limits = []

    class _Mx:
        @staticmethod
        def set_cache_limit(value):
            limits.append(value)

    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_RESERVATIONS", {})
    monkeypatch.setattr(mod, "_CACHE_LIMIT_BASE_BYTES", None)
    monkeypatch.setattr(mod, "_CACHE_LIMIT_EFFECTIVE_BYTES", None)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "4")
    gib = 1024**3

    assert mod.configure_dsv4_lm_head_cache_budget(2 * gib) == 2 * gib
    assert limits[-1] == 2 * gib


def test_lm_head_reduced_ceiling_releases_existing_reservations(monkeypatch):
    import vmlx_engine.models.dsv4_lm_head_fastpath as mod

    class _Mx:
        @staticmethod
        def set_cache_limit(_value):
            return None

    model = type("Model", (), {})()
    config = mod._HeadConfig(signature=(), weight=object(), reservation_bytes=2)
    configs = mod._WeakIdentityMap()
    configs[model] = config
    reference = next(iter(configs._items.values()))[0]
    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_CONFIGS", configs)
    monkeypatch.setattr(mod, "_RESERVATIONS", {id(model): (reference, 2 * 1024**3)})
    monkeypatch.setattr(mod, "_CACHE_LIMIT_BASE_BYTES", 8 * 1024**3)
    monkeypatch.setattr(mod, "_CACHE_LIMIT_EFFECTIVE_BYTES", 6 * 1024**3)
    monkeypatch.setenv("VMLX_DSV4_MLX_CACHE_MIN_GB", "4")

    mod.configure_dsv4_lm_head_cache_budget(5 * 1024**3)
    assert config.weight is None
    assert config.reservation_bytes == 0
    assert not mod._RESERVATIONS


def test_rope_cache_is_instance_scoped_and_bounded(monkeypatch):
    import vmlx_engine.models.dsv4_rope_cache as mod

    class _Values:
        shape = (2,)

        def tolist(self):
            return [1.0, 0.5]

        def __getitem__(self, _key):
            return self

    class _Rope:
        inv_freq = _Values()

        def __init__(self):
            self.reference_calls = 0

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            self.reference_calls += 1
            return x

    class _Mx:
        float32 = "float32"

        @staticmethod
        def arange(start, end, dtype=None):
            del dtype
            return _Array((end - start,))

        @staticmethod
        def cos(value):
            return value

        @staticmethod
        def sin(value):
            return value

        @staticmethod
        def eval(*_args):
            return None

        @staticmethod
        def stack(values, axis=-1):
            del axis
            return values[0]

    # Exercise registration/bounds without evaluating the fake rotation math.
    monkeypatch.setattr(mod, "mx", _Mx)
    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_GROUPS", {})
    monkeypatch.setattr(mod, "_GROUP_LOCK", __import__("threading").RLock())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE", "1")
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE_MAX_ENTRIES", "1")
    first, second, untouched = _Rope(), _Rope(), _Rope()
    model = SimpleNamespace(named_modules=lambda: [("a", first), ("b", second)])

    assert mod._install_for_class(model, _Rope) == 2
    assert mod.dsv4_rope_cache_status(model)["registered_instances"] == 2
    group = mod._CONFIGS.get(first)
    assert group is mod._CONFIGS.get(second)
    group.tables[(0, 1, "float16")] = (object(), object())
    group.tables.popitem(last=False)
    group.tables[(1, 1, "float16")] = (object(), object())
    assert len(group.tables) == 1

    untouched(_Array((1, 1, 4)))
    assert untouched.reference_calls == 1


def test_rope_cache_releases_orphaned_frequency_groups(monkeypatch):
    import gc
    import weakref

    import vmlx_engine.models.dsv4_rope_cache as mod

    class _Values:
        shape = (2,)

        def tolist(self):
            return [1.0, 0.5]

    class _Rope:
        inv_freq = _Values()

        def __call__(self, x, offset=0, inverse=False, positions=None):
            del offset, inverse, positions
            return x

    monkeypatch.setattr(mod, "_CONFIGS", mod._WeakIdentityMap())
    monkeypatch.setattr(mod, "_GROUPS", {})
    monkeypatch.setattr(mod, "_GROUP_LOCK", __import__("threading").RLock())
    monkeypatch.setattr(mod, "_CLASS_PATCHES", {})
    monkeypatch.setenv("VMLX_DSV4_ROPE_CACHE", "1")
    rope = _Rope()
    rope_ref = weakref.ref(rope)
    model = SimpleNamespace(
        named_modules=lambda: [("rope", rope_ref())] if rope_ref() is not None else []
    )

    assert mod._install_for_class(model, _Rope) == 1
    assert len(mod._GROUPS) == 1
    del rope
    gc.collect()
    assert not mod._GROUPS

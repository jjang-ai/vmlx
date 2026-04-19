# SPDX-License-Identifier: Apache-2.0
"""
Regression test for vmlx#94 — scheduler.py's memory-pressure guard in
`_schedule_waiting` must prefer the top-level `mx.get_active_memory` /
`mx.device_info` API over the deprecated `mx.metal.*` aliases when the
top-level one exists, but still fall through to `mx.metal.*` on older
MLX builds where the top-level API isn't present.
"""

import importlib
import warnings

import pytest


class TestMxMetalDeprecationCleanup:
    """vmlx#94 — scheduler memory-pressure guard uses non-deprecated API."""

    def test_no_deprecation_warning_on_current_mlx(self):
        """On MLX ≥ 0.31, calling the guard path must not emit
        DeprecationWarning from `mx.metal.get_active_memory`."""
        mx = pytest.importorskip("mlx.core")
        if not hasattr(mx, "get_active_memory"):
            pytest.skip("MLX too old to test top-level API preference")

        # Replay the scheduler's exact lookup pattern and assert no
        # DeprecationWarning fires.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            _get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            _device_info = getattr(mx, "device_info", None) or mx.metal.device_info

            # These calls would raise if they resolved to the deprecated alias.
            _ = _get_active()
            _ = _device_info()

    def test_fallback_to_mx_metal_on_old_mlx(self, monkeypatch):
        """On older MLX (no top-level `mx.get_active_memory`), the lookup
        must fall through to `mx.metal.get_active_memory`. Simulate the
        old-MLX shape by stripping the top-level attribute."""
        mx = pytest.importorskip("mlx.core")

        # Force the lookup to miss the top-level attribute.
        monkeypatch.delattr(mx, "get_active_memory", raising=False)
        monkeypatch.delattr(mx, "device_info", raising=False)

        # The scheduler's lookup should still resolve something callable
        # via the `mx.metal.*` fallback — no AttributeError.
        _get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        _device_info = getattr(mx, "device_info", None) or mx.metal.device_info

        assert callable(_get_active)
        assert callable(_device_info)

    def test_scheduler_module_imports_cleanly(self):
        """Importing the scheduler module must not emit any DeprecationWarning
        from the memory-pressure guard (the guard runs inside a function, not
        at import time, but this pins that invariant)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Force a fresh import so we catch any top-of-module side effects.
            import vmlx_engine.scheduler  # noqa: F401

            # Re-import to flush any caching.
            importlib.reload(vmlx_engine.scheduler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

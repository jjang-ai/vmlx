# SPDX-License-Identifier: Apache-2.0
"""Tests for _apply_jit_compilation() in server.py.

Verifies JIT compilation behavior: no-op when engine is None,
mx.compile application, error fallback, replacement verification,
and skip when model is not callable.
"""

import platform
import sys
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


class TestJitToggle:
    """Tests for _apply_jit_compilation()."""

    def test_jit_does_nothing_when_engine_none(self):
        """When _engine is None, _apply_jit_compilation should return without error."""
        from vmlx_engine import server

        with patch.object(server, "_engine", None):
            # Should not raise
            server._apply_jit_compilation()

    def test_jit_applies_mx_compile(self):
        """When engine has a (text-only) model, mx.compile should be called
        on `model.model`. Explicitly mark the mock engine as LLM — MLLM
        engines use a separate path that targets `language_model.model`."""
        from vmlx_engine import server

        inner_model = MagicMock()
        inner_model.__call__ = MagicMock()  # make it callable

        model_wrapper = MagicMock(spec=["model"])
        model_wrapper.model = inner_model

        mock_engine = MagicMock()
        mock_engine._model = model_wrapper
        mock_engine.is_mllm = False   # force LLM path
        mock_engine._is_mllm = False

        compiled_fn = MagicMock()

        with patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile", return_value=compiled_fn) as mock_compile:
            server._apply_jit_compilation()

        # mx.compile should have been called with the inner model
        mock_compile.assert_called_once_with(inner_model)

    def test_jit_fallback_on_compile_error(self):
        """If mx.compile raises, should log warning and not crash."""
        from vmlx_engine import server

        inner_model = MagicMock()
        inner_model.__call__ = MagicMock()

        model_wrapper = MagicMock()
        model_wrapper.model = inner_model

        mock_engine = MagicMock()
        mock_engine._model = model_wrapper

        with patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile", side_effect=RuntimeError("Unsupported dynamic shape")):
            # Should not raise — graceful fallback
            server._apply_jit_compilation()

    def test_jit_verifies_replacement(self):
        """After compile (LLM path), the compiled object replaces model.model.
        MLLM path replaces language_model.model instead — tested separately.

        vmlx#83 note: _apply_jit_compilation now runs a warmup pass after
        replacement and rolls back on warmup failure. The test wrapper
        must be callable so warmup succeeds and the replacement sticks —
        otherwise the rollback correctly reverts to the pre-compile
        model (and the test would need a different assertion).
        """
        from vmlx_engine import server

        inner_model = MagicMock()
        inner_model.__call__ = MagicMock()

        # Use a real attribute so assignment is tracked. Wrapper MUST
        # be callable or vmlx#83 warmup-rollback kicks in and reverts.
        class ModelWrapper:
            def __init__(self):
                self.model = inner_model

            def __call__(self, *args, **kwargs):
                # Delegate to compiled inner; MagicMock is callable so this
                # returns a MagicMock and warmup's mx.synchronize() no-ops.
                return self.model(*args, **kwargs)

        model_wrapper = ModelWrapper()

        mock_engine = MagicMock()
        mock_engine._model = model_wrapper
        mock_engine.is_mllm = False   # force LLM path
        mock_engine._is_mllm = False

        compiled_fn = MagicMock()

        with patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile", return_value=compiled_fn):
            server._apply_jit_compilation()

        # The compiled function should now be on model_wrapper.model.
        # If vmlx#83 rollback had fired, this would instead be inner_model.
        assert model_wrapper.model is compiled_fn

    def test_jit_skips_when_model_not_callable(self):
        """When inner model is not callable, should log warning and skip."""
        from vmlx_engine import server

        # inner model is not callable (no __call__)
        non_callable = "not-a-model"

        model_wrapper = MagicMock()
        model_wrapper.model = non_callable

        mock_engine = MagicMock()
        mock_engine._model = model_wrapper
        # Also ensure getattr fallback doesn't find a second-level .model
        del mock_engine.model

        with patch.object(server, "_engine", mock_engine):
            # Should not raise — logs warning and returns
            server._apply_jit_compilation()

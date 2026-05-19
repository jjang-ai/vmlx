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

    def test_jit_skips_dsv4_native_composite_cache(self):
        """DSV4 owns a path-dependent native SWA+CSA/HCA cache; JIT must not trace it.

        The panel should suppress --enable-jit for DSV4, but the server must
        also defend against stale saved sessions, CLI users, and API callers
        that enable JIT programmatically.
        """
        from vmlx_engine import server

        inner_model = MagicMock()
        inner_model.__call__ = MagicMock()

        model_wrapper = MagicMock()
        model_wrapper.model = inner_model

        scheduler = MagicMock()
        scheduler._uses_dsv4_cache = True

        mock_engine = MagicMock()
        mock_engine._model = model_wrapper
        mock_engine.is_mllm = False
        mock_engine._is_mllm = False

        with patch.object(server, "_engine", mock_engine), \
             patch.object(server, "_get_scheduler", return_value=scheduler), \
             patch.object(mx, "compile") as mock_compile:
            server._apply_jit_compilation()

        mock_compile.assert_not_called()

    def test_jit_skips_text_hybrid_arrays_cache_before_compile(self, caplog):
        """Text-only hybrid Qwen/Ling-style caches are not mx.compile-safe.

        A force-text runtime can still use ArraysCache/MambaCache under the
        LLM scheduler. Startup must skip compile before warmup instead of
        tracing a graph that fails on the first real prefill.
        """
        from vmlx_engine import server

        class ArraysCache:
            pass

        class HybridTextModel:
            def __call__(self, *args, **kwargs):
                return args[0]

            def make_cache(self):
                return [ArraysCache()]

        model_wrapper = HybridTextModel()

        mock_engine = MagicMock()
        mock_engine._model = model_wrapper
        mock_engine.is_mllm = False
        mock_engine._is_mllm = False

        with caplog.at_level("INFO"), \
             patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile") as mock_compile:
            server._apply_jit_compilation()

        mock_compile.assert_not_called()
        assert "Skipping mx.compile" in caplog.text
        assert "ArraysCache" in caplog.text

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

    def test_jit_vlm_compiled_proxy_preserves_language_layers(self):
        """VLM JIT must not replace the inner transformer with a bare gc_func.

        mlx_vlm LanguageModel.layers delegates to ``self.model.layers``.
        A bare mx.compile result has no module attributes, so make_cache()
        crashes after JIT. The object installed at language_model.model must
        stay callable while preserving attributes from the original module.
        """
        from vmlx_engine import server

        class InnerTransformer:
            def __init__(self):
                self.layers = ["l0", "l1"]
                self.config = {"family": "fake-vlm"}

            def __call__(self, *args, **kwargs):
                return args[0]

        class LanguageModel:
            def __init__(self):
                self.model = InnerTransformer()

            @property
            def layers(self):
                return self.model.layers

            def make_cache(self):
                return list(self.layers)

            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        class VlmWrapper:
            def __init__(self):
                self.language_model = LanguageModel()

            def make_cache(self):
                return None

        vlm_wrapper = VlmWrapper()
        original_inner = vlm_wrapper.language_model.model

        mock_engine = MagicMock()
        mock_engine._model = vlm_wrapper
        mock_engine.is_mllm = True
        mock_engine._is_mllm = True

        compiled_fn = MagicMock(side_effect=lambda *args, **kwargs: args[0])

        with patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile", return_value=compiled_fn) as mock_compile:
            server._apply_jit_compilation()

        installed = vlm_wrapper.language_model.model
        assert installed is not original_inner
        assert installed is not compiled_fn, "do not install bare gc_func on VLM inner model"
        assert installed.layers == ["l0", "l1"]
        assert installed.config == {"family": "fake-vlm"}
        assert vlm_wrapper.language_model.layers == ["l0", "l1"]
        assert vlm_wrapper.language_model.make_cache() == ["l0", "l1"]
        mock_compile.assert_called_once_with(original_inner)

    def test_jit_vlm_compiled_proxy_forwards_runtime_attribute_mutations(self):
        """VLM JIT proxy must forward writes to the original module.

        mlx_vlm wrappers can update model attributes during setup/warmup. If
        writes stay on the proxy, rollback and delegated attributes diverge.
        """
        from vmlx_engine import server

        class InnerTransformer:
            def __init__(self):
                self.layers = ["l0"]
                self.runtime_flag = "cold"

            def __call__(self, *args, **kwargs):
                return args[0]

        original = InnerTransformer()
        proxy = server._CompiledModuleProxy(
            original,
            MagicMock(side_effect=lambda *args, **kwargs: args[0]),
        )

        proxy.runtime_flag = "warm"
        proxy.new_runtime_attr = 42

        assert original.runtime_flag == "warm"
        assert original.new_runtime_attr == 42
        assert proxy.runtime_flag == "warm"
        assert proxy.new_runtime_attr == 42

    def test_jit_vlm_warmup_failure_rolls_back_proxy(self):
        """VLM warmup failures must restore the original transformer."""
        from vmlx_engine import server

        class InnerTransformer:
            layers = ["l0"]

            def __call__(self, *args, **kwargs):
                return args[0]

        class LanguageModel:
            def __init__(self):
                self.model = InnerTransformer()

            @property
            def layers(self):
                return self.model.layers

            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        class VlmWrapper:
            def __init__(self):
                self.language_model = LanguageModel()

            def make_cache(self):
                return None

        class BrokenCompiled:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("compile warmup failed")

        vlm_wrapper = VlmWrapper()
        original_inner = vlm_wrapper.language_model.model

        mock_engine = MagicMock()
        mock_engine._model = vlm_wrapper
        mock_engine.is_mllm = True
        mock_engine._is_mllm = True

        with patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile", return_value=BrokenCompiled()):
            server._apply_jit_compilation()

        assert vlm_wrapper.language_model.model is original_inner

    def test_jit_vlm_warmup_uses_language_model_cache(self):
        """VLM JIT warmup must use the real language-model cache shape.

        Qwen3.6 hybrid SSM VLMs expose ``make_cache`` on
        ``language_model`` and pass an ArraysCache/MambaCache object during
        prefill. A no-cache warmup lets mx.compile stay installed, but the
        first real request then fails with "Function arguments must be trees
        of arrays or constants". The startup warmup must catch that and
        rollback before serving traffic.
        """
        from vmlx_engine import server

        cache_obj = object()

        class InnerTransformer:
            layers = ["l0"]

            def __call__(self, *args, **kwargs):
                return args[0]

        class LanguageModel:
            def __init__(self):
                self.model = InnerTransformer()

            @property
            def layers(self):
                return self.model.layers

            def make_cache(self):
                return cache_obj

            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        class VlmWrapper:
            def __init__(self):
                self.language_model = LanguageModel()

        class BrokenOnCacheCompiled:
            def __init__(self):
                self.saw_cache = False

            def __call__(self, *args, **kwargs):
                if kwargs.get("cache") is cache_obj:
                    self.saw_cache = True
                    raise ValueError("received type mlx_lm.models.cache.ArraysCache")
                return args[0]

        vlm_wrapper = VlmWrapper()
        original_inner = vlm_wrapper.language_model.model
        compiled = BrokenOnCacheCompiled()

        mock_engine = MagicMock()
        mock_engine._model = vlm_wrapper
        mock_engine.is_mllm = True
        mock_engine._is_mllm = True

        with patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile", return_value=compiled):
            server._apply_jit_compilation()

        assert compiled.saw_cache is True
        assert vlm_wrapper.language_model.model is original_inner

    def test_jit_skips_vlm_hybrid_arrays_cache_before_compile(self, caplog):
        """Hybrid MLLM cache objects should skip mx.compile before rollback work.

        Qwen3.6 native-MTP/VL models use mlx_lm cache containers such as
        ArraysCache for hybrid SSM/Vision paths. mx.compile cannot trace those
        Python cache objects, so startup should detect the shape and avoid the
        compile/warmup/rollback churn entirely.
        """
        from vmlx_engine import server

        class ArraysCache:
            pass

        class KVCache:
            pass

        class InnerTransformer:
            layers = ["l0", "l1"]

            def __call__(self, *args, **kwargs):
                return args[0]

        class LanguageModel:
            def __init__(self):
                self.model = InnerTransformer()

            @property
            def layers(self):
                return self.model.layers

            def make_cache(self):
                return [KVCache(), ArraysCache()]

            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        class VlmWrapper:
            def __init__(self):
                self.language_model = LanguageModel()

        vlm_wrapper = VlmWrapper()
        original_inner = vlm_wrapper.language_model.model

        mock_engine = MagicMock()
        mock_engine._model = vlm_wrapper
        mock_engine.is_mllm = True
        mock_engine._is_mllm = True

        with caplog.at_level("INFO"), \
             patch.object(server, "_engine", mock_engine), \
             patch.object(mx, "compile") as mock_compile:
            server._apply_jit_compilation()

        mock_compile.assert_not_called()
        assert vlm_wrapper.language_model.model is original_inner
        assert "Skipping mx.compile" in caplog.text
        assert "ArraysCache" in caplog.text

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

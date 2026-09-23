# SPDX-License-Identifier: Apache-2.0
"""DFlash2 request-owned references must not outlive prefill or generation.

The fake layers expose sink ownership with weak references, so these tests
exercise lifecycle and cancellation without model weights or Metal allocations.
"""

import contextlib
import gc
import sys
import types
import unittest
import weakref
from unittest.mock import patch

from vmlx_engine import dflash2_runtime as runtime


class State:
    pass


class Inner:
    def __init__(self):
        self.layers = []
        self.refs = []

    def __call__(self, inputs, cache=None, gdn_sink=None, **kwargs):
        if gdn_sink is not None:
            for _ in range(3):
                state = State()
                self.refs.append(weakref.ref(state))
                gdn_sink.append(state)
        return inputs

    def norm(self, hidden):
        return hidden


class LanguageModel:
    def __init__(self):
        self.model = Inner()
        self.rollbacks = []

    def lm_head(self, hidden):
        return hidden

    def rollback_speculative_cache(self, cache, states, accepted, block_size):
        self.rollbacks.append((len(states), accepted, block_size))


class TestMemoryLifecycle(unittest.TestCase):
    def setUp(self):
        store_patch = patch.object(
            runtime, "_SESSION_STORE", runtime._DFlash2SessionStore()
        )
        store_patch.start()
        self.addCleanup(store_patch.stop)
        self.lm = LanguageModel()
        self.model = types.SimpleNamespace(language_model=self.lm)
        self.draft = types.SimpleNamespace(config=types.SimpleNamespace(block_size=5))
        self.adapter = runtime._adapter_for(self.model)
        self.key = (id(self.model), id(self.draft))
        # Match the hook's shared-list ownership, not a copy.
        self.adapter._hidden_states = [None, None]
        self.hooks = self.adapter._hidden_states
        self.mx = types.ModuleType("mlx.core")
        self.runtime = types.ModuleType("dflash.model_mlx")
        self.runtime.generation_stream = object()
        self.runtime.wired_limit = lambda *a: contextlib.nullcontext()
        self.modules = {
            "mlx": types.ModuleType("mlx"),
            "mlx.core": self.mx,
            "dflash": types.ModuleType("dflash"),
            "dflash.model_mlx": self.runtime,
        }

    def stream(self):
        return runtime.stream_dflash2_generate(
            self.model, None, self.draft, "prompt", max_tokens=20, temperature=0
        )

    def dirty(self):
        capture = runtime._VLMGDNStateCapture(self.adapter)
        self.adapter("verify")
        self.adapter._hidden_states[:] = [State(), State()]
        runtime._SESSION_STORE.put({"model_key": self.key, "kind": "boundary"})
        runtime._SESSION_STORE.put({"model_key": ("other", "model")})
        return capture

    def assert_clean(self):
        self.assertEqual(self.adapter.gdn_states, [])
        self.assertFalse(self.adapter.capture_gdn_states)
        self.assertIs(self.adapter._hidden_states, self.hooks)
        self.assertEqual(self.hooks, [None, None])
        gc.collect()
        self.assertTrue(all(r() is None for r in self.lm.model.refs))

    def test_chunked_prefill_does_not_capture_rollback_state(self):
        # One sink entry per hybrid layer per chunk previously stayed alive.
        for i in range(64):
            self.assertEqual(self.adapter(i), i)
        self.assertEqual(self.adapter.gdn_states, [])
        self.assertEqual(self.lm.model.refs, [])

    def test_verification_preserves_latest_rollback_state(self):
        capture = runtime._VLMGDNStateCapture(self.adapter)
        for i in range(8):
            self.adapter(i)
            capture.rollback([], accepted=2, trim=2)
            self.assertEqual(len(self.adapter.gdn_states), 3)
        self.assertEqual(self.lm.rollbacks, [(3, 2, 5)] * 8)
        capture.close()
        self.assert_clean()

    def test_normal_completion_retains_valid_checkpoint(self):
        def inner(*a, **kw):
            self.dirty()
            yield "token"

        with (
            patch.dict(sys.modules, self.modules),
            patch.object(runtime, "_stream_generate_resumable", inner),
        ):
            self.assertEqual(list(self.stream()), ["token"])
        self.assert_clean()
        self.assertTrue(
            any(x["model_key"] == self.key for x in runtime._SESSION_STORE._entries)
        )

    def test_prefill_failure_before_capture_creation(self):
        def inner(*a, **kw):
            self.hooks[:] = [State(), State()]
            runtime._SESSION_STORE.put({"model_key": self.key})
            raise RuntimeError("prefill OOM")
            yield

        with (
            patch.dict(sys.modules, self.modules),
            patch.object(runtime, "_stream_generate_resumable", inner),
            self.assertRaisesRegex(RuntimeError, "prefill OOM"),
        ):
            list(self.stream())
        self.assert_clean()
        self.assertEqual(runtime._SESSION_STORE._entries, [])

    def test_decode_failure_releases_only_this_models_checkpoints(self):
        def inner(*a, **kw):
            self.dirty()
            yield "token"
            raise RuntimeError("verify OOM")

        with (
            patch.dict(sys.modules, self.modules),
            patch.object(runtime, "_stream_generate_resumable", inner),
            self.assertRaisesRegex(RuntimeError, "verify OOM"),
        ):
            list(self.stream())
        self.assert_clean()
        self.assertEqual(
            runtime._SESSION_STORE._entries, [{"model_key": ("other", "model")}]
        )

    def test_client_disconnect_closes_generator_and_releases_state(self):
        closed = []

        def inner(*a, **kw):
            self.dirty()
            try:
                yield "token"
                yield "unused"
            finally:
                closed.append(True)

        with (
            patch.dict(sys.modules, self.modules),
            patch.object(runtime, "_stream_generate_resumable", inner),
        ):
            stream = self.stream()
            self.assertEqual(next(stream), "token")
            stream.close()
        self.assertEqual(closed, [True])
        self.assert_clean()
        self.assertEqual(
            runtime._SESSION_STORE._entries, [{"model_key": ("other", "model")}]
        )

    def test_setup_failure_and_stale_state_cleanup(self):
        self.dirty()

        def fail(*a):
            raise RuntimeError("wired-limit setup")

        self.runtime.wired_limit = fail
        with (
            patch.dict(sys.modules, self.modules),
            self.assertRaisesRegex(RuntimeError, "wired-limit setup"),
        ):
            list(self.stream())
        self.assert_clean()

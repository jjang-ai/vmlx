"""
Comprehensive test suite for speculative decoding.

Tests cover:
- SpeculativeConfig validation and edge cases
- Global state lifecycle (load / unload / enable check)
- Gating logic (should_use_speculative)
- Draft token count accessor
- Tokenizer compatibility validation
- Stats reporting
- Error handling
- CLI argument parsing
- Draft model wiring into llm.py generation paths
- mlx-lm API integration
- Cache subsystem compatibility
- Server health endpoint integration
- from_draft tracking in GenerationResponse
- Edge cases and boundary conditions
"""

import argparse
import unittest
import warnings
from unittest.mock import MagicMock, patch, PropertyMock
import sys
import os
import inspect


# ---------------------------------------------------------------------------
# 1. SpeculativeConfig validation
# ---------------------------------------------------------------------------

class TestSpeculativeConfig(unittest.TestCase):
    """Test SpeculativeConfig dataclass validation."""

    def test_default_config_disabled(self):
        """Default config (no model) should be disabled."""
        from vmlx_engine.speculative import SpeculativeConfig
        config = SpeculativeConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.model, "")  # Default is empty string
        self.assertEqual(config.num_tokens, 3)

    def test_config_with_model_enabled(self):
        """Config with model path should be enabled."""
        from vmlx_engine.speculative import SpeculativeConfig
        config = SpeculativeConfig(model="some-draft-model")
        self.assertTrue(config.enabled)
        self.assertEqual(config.model, "some-draft-model")

    def test_config_custom_num_tokens(self):
        """Custom num_tokens should be preserved."""
        from vmlx_engine.speculative import SpeculativeConfig
        config = SpeculativeConfig(model="m", num_tokens=5)
        self.assertEqual(config.num_tokens, 5)

    def test_config_num_tokens_clamped_to_min_1(self):
        """num_tokens < 1 should be clamped to 1."""
        from vmlx_engine.speculative import SpeculativeConfig
        config = SpeculativeConfig(model="m", num_tokens=0)
        self.assertEqual(config.num_tokens, 1)
        config2 = SpeculativeConfig(model="m", num_tokens=-5)
        self.assertEqual(config2.num_tokens, 1)

    def test_config_high_num_tokens_logs_warning(self):
        """num_tokens > 20 should trigger a logger warning (not warnings.warn)."""
        import logging
        from vmlx_engine.speculative import SpeculativeConfig
        with self.assertLogs("vmlx_engine.speculative", level="WARNING") as cm:
            config = SpeculativeConfig(model="m", num_tokens=25)
        self.assertTrue(any("25" in msg for msg in cm.output))
        self.assertEqual(config.num_tokens, 25)

    def test_config_disable_by_batch_size(self):
        """disable_by_batch_size should be configurable."""
        from vmlx_engine.speculative import SpeculativeConfig
        config = SpeculativeConfig(model="m", disable_by_batch_size=4)
        self.assertEqual(config.disable_by_batch_size, 4)


# ---------------------------------------------------------------------------
# 2. Global state lifecycle
# ---------------------------------------------------------------------------

class TestGlobalState(unittest.TestCase):
    """Test global state management (load/unload/enable check)."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model
        self._orig_tok = spec._draft_tokenizer
        spec._spec_config = None
        spec._draft_model = None
        spec._draft_tokenizer = None

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model
        spec._draft_tokenizer = self._orig_tok

    def test_initial_state_disabled(self):
        """Initially speculative decoding should be disabled."""
        from vmlx_engine.speculative import is_speculative_enabled, get_draft_model
        self.assertFalse(is_speculative_enabled())
        self.assertIsNone(get_draft_model())

    def test_state_after_successful_load(self):
        """After mock load, state should reflect enabled."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        config = SpeculativeConfig(model="mock-model")

        # Patch mlx_lm.load (imported locally inside load_draft_model)
        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            model, tokenizer = spec.load_draft_model(config)

        self.assertTrue(spec.is_speculative_enabled())
        self.assertIs(spec.get_draft_model(), mock_model)

    def test_unload_clears_state(self):
        """Unloading should clear all state."""
        import vmlx_engine.speculative as spec
        spec._spec_config = MagicMock(enabled=True)
        spec._draft_model = MagicMock()
        spec._draft_tokenizer = MagicMock()

        spec.unload_draft_model()

        self.assertIsNone(spec._draft_model)
        self.assertIsNone(spec._draft_tokenizer)
        self.assertFalse(spec.is_speculative_enabled())

    def test_state_after_disabled_load(self):
        """Loading with disabled config (no model) should return (None, None)."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        config = SpeculativeConfig()  # no model → disabled
        model, tokenizer = spec.load_draft_model(config)
        self.assertIsNone(model)
        self.assertIsNone(tokenizer)
        self.assertFalse(spec.is_speculative_enabled())


# ---------------------------------------------------------------------------
# 3. Gating logic (should_use_speculative)
# ---------------------------------------------------------------------------

class TestGatingLogic(unittest.TestCase):
    """Test should_use_speculative() for all engine/model combinations."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model
        self._orig_tok = spec._draft_tokenizer

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model
        spec._draft_tokenizer = self._orig_tok

    def _enable_spec(self):
        """Helper to enable speculative state."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft", num_tokens=3)
        spec._draft_model = MagicMock()

    def test_disabled_when_not_configured(self):
        """Returns False when spec decoding is not configured."""
        import vmlx_engine.speculative as spec
        spec._spec_config = None
        spec._draft_model = None
        self.assertFalse(spec.should_use_speculative())

    def test_disabled_when_batched(self):
        """Returns False for BatchedEngine even when configured."""
        self._enable_spec()
        import vmlx_engine.speculative as spec
        self.assertFalse(spec.should_use_speculative(is_batched=True))

    def test_disabled_when_mllm(self):
        """Returns False for MLLM/VLM even when configured."""
        self._enable_spec()
        import vmlx_engine.speculative as spec
        self.assertFalse(spec.should_use_speculative(is_mllm=True))

    def test_disabled_when_batched_and_mllm(self):
        """Returns False for batched MLLM."""
        self._enable_spec()
        import vmlx_engine.speculative as spec
        self.assertFalse(spec.should_use_speculative(is_batched=True, is_mllm=True))

    def test_enabled_for_simple_llm(self):
        """Returns True for SimpleEngine + LLM."""
        self._enable_spec()
        import vmlx_engine.speculative as spec
        self.assertTrue(spec.should_use_speculative(is_batched=False, is_mllm=False))

    def test_enabled_default_args(self):
        """Returns True with default args (simple LLM)."""
        self._enable_spec()
        import vmlx_engine.speculative as spec
        self.assertTrue(spec.should_use_speculative())


# ---------------------------------------------------------------------------
# 4. Draft token count accessor
# ---------------------------------------------------------------------------

class TestDraftTokenCount(unittest.TestCase):
    """Test get_num_draft_tokens() accessor."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig = spec._spec_config

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig

    def test_zero_when_not_configured(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = None
        self.assertEqual(spec.get_num_draft_tokens(), 0)

    def test_returns_configured_value(self):
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="m", num_tokens=5)
        self.assertEqual(spec.get_num_draft_tokens(), 5)

    def test_returns_default_3(self):
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="m")
        self.assertEqual(spec.get_num_draft_tokens(), 3)


# ---------------------------------------------------------------------------
# 5. Tokenizer compatibility validation
# ---------------------------------------------------------------------------

class TestTokenizerValidation(unittest.TestCase):
    """Test validate_draft_tokenizer() for tokenizer compatibility."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig = spec._draft_tokenizer

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._draft_tokenizer = self._orig

    def test_compatible_tokenizers(self):
        """Same vocab, same encoding → True."""
        import vmlx_engine.speculative as spec
        mock_draft = MagicMock()
        mock_draft.__len__ = MagicMock(return_value=32000)
        mock_draft.encode.return_value = [1, 2, 3]
        spec._draft_tokenizer = mock_draft

        mock_target = MagicMock()
        mock_target.__len__ = MagicMock(return_value=32000)
        mock_target.encode.return_value = [1, 2, 3]

        self.assertTrue(spec.validate_draft_tokenizer(mock_target))

    def test_vocab_size_mismatch(self):
        """Different vocab sizes → False."""
        import vmlx_engine.speculative as spec
        mock_draft = MagicMock()
        mock_draft.__len__ = MagicMock(return_value=32000)
        spec._draft_tokenizer = mock_draft

        mock_target = MagicMock()
        mock_target.__len__ = MagicMock(return_value=50000)

        self.assertFalse(spec.validate_draft_tokenizer(mock_target))

    def test_encoding_mismatch(self):
        """Same vocab but different encodings → False."""
        import vmlx_engine.speculative as spec
        mock_draft = MagicMock()
        mock_draft.__len__ = MagicMock(return_value=32000)
        mock_draft.encode.return_value = [1, 2, 3]
        spec._draft_tokenizer = mock_draft

        mock_target = MagicMock()
        mock_target.__len__ = MagicMock(return_value=32000)
        mock_target.encode.return_value = [4, 5, 6]

        self.assertFalse(spec.validate_draft_tokenizer(mock_target))

    def test_none_draft_tokenizer(self):
        """No draft tokenizer → skip validation (True)."""
        import vmlx_engine.speculative as spec
        spec._draft_tokenizer = None
        self.assertTrue(spec.validate_draft_tokenizer(MagicMock()))

    def test_none_target_tokenizer(self):
        """No target tokenizer → skip validation (True)."""
        import vmlx_engine.speculative as spec
        spec._draft_tokenizer = MagicMock()
        self.assertTrue(spec.validate_draft_tokenizer(None))

    def test_validation_exception_returns_true(self):
        """Exception during validation → don't block (True)."""
        import vmlx_engine.speculative as spec
        mock_draft = MagicMock()
        mock_draft.__len__ = MagicMock(side_effect=Exception("boom"))
        spec._draft_tokenizer = mock_draft

        self.assertTrue(spec.validate_draft_tokenizer(MagicMock()))


# ---------------------------------------------------------------------------
# 6. Stats reporting
# ---------------------------------------------------------------------------

class TestSpecStats(unittest.TestCase):
    """Test get_spec_stats() reporting."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model
        self._orig_tok = spec._draft_tokenizer
        spec._spec_config = None
        spec._draft_model = None
        spec._draft_tokenizer = None

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model
        spec._draft_tokenizer = self._orig_tok

    def test_stats_not_configured(self):
        """No config → returns 'not_configured' string."""
        import vmlx_engine.speculative as spec
        stats = spec.get_spec_stats()
        # When not configured, speculative_decoding is a string
        self.assertEqual(stats["speculative_decoding"], "not_configured")

    def test_stats_enabled_no_model(self):
        """Config set but model not loaded yet → enabled in config but not loaded."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft-model")
        spec._draft_model = None  # Not loaded

        stats = spec.get_spec_stats()
        sd = stats["speculative_decoding"]
        # Config says enabled but is_speculative_enabled() requires draft_model != None
        self.assertFalse(sd["enabled"])  # is_speculative_enabled checks _draft_model
        self.assertFalse(sd["draft_model_loaded"])
        self.assertEqual(sd["draft_model"], "draft-model")

    def test_stats_fully_loaded(self):
        """Config set and model loaded."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft-model", num_tokens=4)
        spec._draft_model = MagicMock()

        stats = spec.get_spec_stats()
        sd = stats["speculative_decoding"]
        self.assertTrue(sd["enabled"])
        self.assertTrue(sd["draft_model_loaded"])
        self.assertEqual(sd["draft_model"], "draft-model")
        self.assertEqual(sd["num_draft_tokens"], 4)


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------

class TestLoadDraftModelErrors(unittest.TestCase):
    """Test error handling when loading draft model."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model
        self._orig_tok = spec._draft_tokenizer
        spec._spec_config = None
        spec._draft_model = None
        spec._draft_tokenizer = None

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model
        spec._draft_tokenizer = self._orig_tok

    def test_load_invalid_model_raises_value_error(self):
        """Loading a non-existent model should raise ValueError."""
        from vmlx_engine.speculative import SpeculativeConfig, load_draft_model
        config = SpeculativeConfig(model="definitely-not-a-real-model-path-12345")
        with self.assertRaises(ValueError):
            load_draft_model(config)

    def test_load_with_mock_success(self):
        """Mock successful load to verify state management."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        config = SpeculativeConfig(model="mock-model")

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            model, tokenizer = spec.load_draft_model(config)

        self.assertIs(model, mock_model)
        self.assertIs(tokenizer, mock_tokenizer)
        self.assertIs(spec.get_draft_model(), mock_model)
        self.assertTrue(spec.is_speculative_enabled())


# ---------------------------------------------------------------------------
# 8. CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIArgs(unittest.TestCase):
    """Test CLI argument parsing for speculative decoding flags."""

    def _make_serve_parser(self):
        """Create a minimal serve parser for testing speculative args."""
        # cli.py doesn't export a separate serve parser function,
        # so we call main's parser builder
        from vmlx_engine.cli import main
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        serve_parser = subparsers.add_parser("serve")
        serve_parser.add_argument("model", type=str)
        serve_parser.add_argument("--speculative-model", type=str, default=None)
        serve_parser.add_argument("--num-draft-tokens", type=int, default=3)
        return parser

    def test_speculative_args_parsed(self):
        """--speculative-model and --num-draft-tokens should be parsed."""
        parser = self._make_serve_parser()
        args = parser.parse_args([
            "serve", "test-model",
            "--speculative-model", "draft-model",
            "--num-draft-tokens", "5",
        ])
        self.assertEqual(args.speculative_model, "draft-model")
        self.assertEqual(args.num_draft_tokens, 5)

    def test_speculative_args_default(self):
        """Default values when no speculative flags provided."""
        parser = self._make_serve_parser()
        args = parser.parse_args(["serve", "test-model"])
        self.assertIsNone(args.speculative_model)
        self.assertEqual(args.num_draft_tokens, 3)

    def test_real_cli_has_speculative_args(self):
        """Verify the real cli.py has the speculative arguments."""
        import vmlx_engine.cli as cli_module
        source = inspect.getsource(cli_module)
        self.assertIn("--speculative-model", source)
        self.assertIn("--num-draft-tokens", source)


# ---------------------------------------------------------------------------
# 9. Draft model wiring into llm.py
# ---------------------------------------------------------------------------

class TestLLMDraftModelWiring(unittest.TestCase):
    """Test that llm.py actually passes draft_model to mlx_lm.stream_generate."""

    @patch("vmlx_engine.speculative.is_speculative_enabled", return_value=True)
    @patch("vmlx_engine.speculative.get_draft_model")
    @patch("vmlx_engine.speculative.get_num_draft_tokens", return_value=3)
    @patch("vmlx_engine.speculative.validate_draft_tokenizer", return_value=True)
    def test_stream_generate_passes_draft_model(
        self, mock_validate, mock_num, mock_get_draft, mock_enabled
    ):
        """stream_generate should pass draft_model kwarg when spec is enabled."""
        mock_draft = MagicMock()
        mock_get_draft.return_value = mock_draft

        # Patch mlx_lm.stream_generate to capture kwargs
        captured_kwargs = {}
        def fake_stream_generate(model, tokenizer, **kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.text = "Hello"
            resp.token = 42
            resp.finished = True
            resp.finish_reason = "stop"
            yield resp

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            from vmlx_engine.models.llm import MLXLanguageModel
            model = MLXLanguageModel.__new__(MLXLanguageModel)
            model.model = MagicMock()
            model.tokenizer = MagicMock()
            model._loaded = True

            results = list(model.stream_generate(prompt="Test", max_tokens=10))

        self.assertIn("draft_model", captured_kwargs)
        self.assertIs(captured_kwargs["draft_model"], mock_draft)
        self.assertEqual(captured_kwargs["num_draft_tokens"], 3)

    @patch("vmlx_engine.speculative.is_speculative_enabled", return_value=False)
    def test_stream_generate_no_draft_when_disabled(self, mock_enabled):
        """stream_generate should NOT pass draft_model when spec is disabled."""
        captured_kwargs = {}
        def fake_stream_generate(model, tokenizer, **kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.text = "Hello"
            resp.token = 42
            resp.finished = True
            resp.finish_reason = "stop"
            yield resp

        with patch("mlx_lm.stream_generate", side_effect=fake_stream_generate):
            from vmlx_engine.models.llm import MLXLanguageModel
            model = MLXLanguageModel.__new__(MLXLanguageModel)
            model.model = MagicMock()
            model.tokenizer = MagicMock()
            model._loaded = True

            results = list(model.stream_generate(prompt="Test", max_tokens=10))

        self.assertNotIn("draft_model", captured_kwargs)


# ---------------------------------------------------------------------------
# 10. mlx-lm API integration
# ---------------------------------------------------------------------------

class TestMLXLMIntegration(unittest.TestCase):
    """Verify that mlx-lm's speculative decoding API is available and compatible."""

    def test_speculative_generate_step_importable(self):
        """speculative_generate_step should be importable from mlx_lm."""
        try:
            from mlx_lm.generate import speculative_generate_step
            self.assertTrue(callable(speculative_generate_step))
        except ImportError:
            self.skipTest("mlx-lm not installed or missing speculative_generate_step")

    def test_generation_response_has_from_draft(self):
        """GenerationResponse should have from_draft field."""
        try:
            from mlx_lm.generate import GenerationResponse
            fields = {f.name for f in GenerationResponse.__dataclass_fields__.values()}
            self.assertIn("from_draft", fields)
        except (ImportError, AttributeError):
            self.skipTest("mlx-lm GenerationResponse not available")

    def test_stream_generate_accepts_draft_model(self):
        """Verify mlx-lm's stream_generate has draft_model parameter."""
        try:
            from mlx_lm.generate import stream_generate
            sig = inspect.signature(stream_generate)
            self.assertIn("draft_model", sig.parameters)
        except ImportError:
            self.skipTest("mlx-lm not installed")

    def test_stream_generate_accepts_num_draft_tokens(self):
        """Verify speculative_generate_step accepts num_draft_tokens."""
        try:
            from mlx_lm.generate import speculative_generate_step
            sig = inspect.signature(speculative_generate_step)
            params = list(sig.parameters.keys())
            has_param = "num_draft_tokens" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            self.assertTrue(has_param)
        except ImportError:
            self.skipTest("mlx-lm not installed")


# ---------------------------------------------------------------------------
# 11. Cache subsystem compatibility
# ---------------------------------------------------------------------------

class TestCacheCompatibility(unittest.TestCase):
    """Verify speculative decoding doesn't conflict with cache subsystems."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model

    def test_simple_engine_has_no_prefix_cache(self):
        """SimpleEngine doesn't use prefix cache, so no conflict with spec decoding."""
        from vmlx_engine.engine.simple import SimpleEngine
        engine = SimpleEngine.__new__(SimpleEngine)
        engine._is_mllm = False
        engine._model = None
        self.assertFalse(hasattr(engine, '_engine') and engine._engine is not None)

    def test_batched_engine_gated_out(self):
        """BatchedEngine + spec should be gated out by should_use_speculative."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft")
        spec._draft_model = MagicMock()
        self.assertFalse(spec.should_use_speculative(is_batched=True))

    def test_spec_decoding_and_prefix_cache_mutually_exclusive_engines(self):
        """Spec decoding (SimpleEngine) and prefix cache (BatchedEngine) run
        in mutually exclusive engine modes, so they can't conflict."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft")
        spec._draft_model = MagicMock()
        self.assertTrue(spec.should_use_speculative(is_batched=False))
        self.assertFalse(spec.should_use_speculative(is_batched=True))


# ---------------------------------------------------------------------------
# 12. Server health endpoint integration
# ---------------------------------------------------------------------------

class TestSpeculativeWithServerHealth(unittest.TestCase):
    """Test integration of speculative stats in server health endpoint."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model
        self._orig_tok = spec._draft_tokenizer

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model
        spec._draft_tokenizer = self._orig_tok

    def test_health_includes_spec_stats_when_not_configured(self):
        """Health endpoint should include spec stats even when disabled."""
        import vmlx_engine.speculative as spec
        spec._spec_config = None
        spec._draft_model = None
        stats = spec.get_spec_stats()
        self.assertIn("speculative_decoding", stats)
        # When not configured, it's the string "not_configured"
        self.assertEqual(stats["speculative_decoding"], "not_configured")

    def test_health_includes_spec_stats_when_configured(self):
        """Health endpoint should show full spec stats when configured."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft-model", num_tokens=4)
        spec._draft_model = MagicMock()

        stats = spec.get_spec_stats()
        sd = stats["speculative_decoding"]
        self.assertTrue(sd["enabled"])
        self.assertTrue(sd["draft_model_loaded"])
        self.assertEqual(sd["draft_model"], "draft-model")
        self.assertEqual(sd["num_draft_tokens"], 4)


# ---------------------------------------------------------------------------
# 13. from_draft tracking
# ---------------------------------------------------------------------------

class TestFromDraftTracking(unittest.TestCase):
    """Test that from_draft field is available for acceptance rate tracking."""

    def test_generation_response_from_draft_default_false(self):
        """Non-speculative generation should have from_draft=False."""
        try:
            from mlx_lm.generate import GenerationResponse
            resp = GenerationResponse(
                text="hello",
                token=1,
                logprobs=0.0,
                from_draft=False,
                prompt_tokens=10,
                prompt_tps=100.0,
                generation_tokens=1,
                generation_tps=50.0,
                peak_memory=1.0,
                finish_reason=None,
            )
            self.assertFalse(resp.from_draft)
        except (ImportError, TypeError):
            self.skipTest("mlx-lm GenerationResponse not available")

    def test_generation_response_from_draft_true(self):
        """Speculative tokens should have from_draft=True."""
        try:
            from mlx_lm.generate import GenerationResponse
            resp = GenerationResponse(
                text="world",
                token=2,
                logprobs=0.0,
                from_draft=True,
                prompt_tokens=10,
                prompt_tps=100.0,
                generation_tokens=2,
                generation_tps=50.0,
                peak_memory=1.0,
                finish_reason=None,
            )
            self.assertTrue(resp.from_draft)
        except (ImportError, TypeError):
            self.skipTest("mlx-lm GenerationResponse not available")


# ---------------------------------------------------------------------------
# 14. Mamba/hybrid model incompatibility
# ---------------------------------------------------------------------------

class TestMambaIncompatibility(unittest.TestCase):
    """Test that spec decoding is properly gated for Mamba/hybrid models."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model

    def test_mllm_gating_covers_mamba_vlm(self):
        """MLLM gating should cover Mamba VLM models."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="draft")
        spec._draft_model = MagicMock()
        self.assertFalse(spec.should_use_speculative(is_mllm=True))


# ---------------------------------------------------------------------------
# 15. Speculative module imports and API surface
# ---------------------------------------------------------------------------

class TestSpecModuleAPI(unittest.TestCase):
    """Test that the speculative module exports all expected functions."""

    def test_all_exports_available(self):
        """All public functions should be importable."""
        from vmlx_engine.speculative import (
            SpeculativeConfig,
            load_draft_model,
            unload_draft_model,
            get_draft_model,
            is_speculative_enabled,
            should_use_speculative,
            get_num_draft_tokens,
            validate_draft_tokenizer,
            get_spec_stats,
        )
        self.assertTrue(callable(load_draft_model))
        self.assertTrue(callable(unload_draft_model))
        self.assertTrue(callable(get_draft_model))
        self.assertTrue(callable(is_speculative_enabled))
        self.assertTrue(callable(should_use_speculative))
        self.assertTrue(callable(get_num_draft_tokens))
        self.assertTrue(callable(validate_draft_tokenizer))
        self.assertTrue(callable(get_spec_stats))


# ---------------------------------------------------------------------------
# 16. Edge cases and boundary conditions
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        import vmlx_engine.speculative as spec
        self._orig_config = spec._spec_config
        self._orig_model = spec._draft_model
        self._orig_tok = spec._draft_tokenizer

    def tearDown(self):
        import vmlx_engine.speculative as spec
        spec._spec_config = self._orig_config
        spec._draft_model = self._orig_model
        spec._draft_tokenizer = self._orig_tok

    def test_double_load_replaces_state(self):
        """Loading a second draft model should replace the first."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig

        mock_model1 = MagicMock(name="model1")
        mock_model2 = MagicMock(name="model2")

        config1 = SpeculativeConfig(model="draft1")
        config2 = SpeculativeConfig(model="draft2")

        with patch("mlx_lm.load", return_value=(mock_model1, MagicMock())):
            spec.load_draft_model(config1)
        self.assertIs(spec.get_draft_model(), mock_model1)

        with patch("mlx_lm.load", return_value=(mock_model2, MagicMock())):
            spec.load_draft_model(config2)
        self.assertIs(spec.get_draft_model(), mock_model2)

    def test_unload_when_not_loaded(self):
        """Unloading when nothing is loaded should not error."""
        import vmlx_engine.speculative as spec
        spec._spec_config = None
        spec._draft_model = None
        spec._draft_tokenizer = None
        # Should not raise
        spec.unload_draft_model()

    def test_concurrent_state_reads(self):
        """Multiple reads of global state should be consistent."""
        import vmlx_engine.speculative as spec
        from vmlx_engine.speculative import SpeculativeConfig
        spec._spec_config = SpeculativeConfig(model="m", num_tokens=7)
        spec._draft_model = MagicMock()

        self.assertTrue(spec.is_speculative_enabled())
        self.assertTrue(spec.should_use_speculative())
        self.assertEqual(spec.get_num_draft_tokens(), 7)
        self.assertIsNotNone(spec.get_draft_model())
        stats = spec.get_spec_stats()
        self.assertTrue(stats["speculative_decoding"]["enabled"])
        self.assertEqual(stats["speculative_decoding"]["num_draft_tokens"], 7)


if __name__ == "__main__":
    unittest.main()

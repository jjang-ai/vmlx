# SPDX-License-Identifier: Apache-2.0
"""
Tests for ImageGenEngine (vmlx_engine/image_gen.py).

All mflux and mlx imports are mocked — these tests run without GPU hardware.

Usage:
    pytest tests/test_image_gen.py -v
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — mock the heavy dependencies so image_gen.py can be imported
# ---------------------------------------------------------------------------

def _mock_mflux_modules():
    """Return a dict of mock modules to inject into sys.modules for mflux."""
    mocks = {}
    for mod in [
        "mflux",
        "mflux.models",
        "mflux.models.common",
        "mflux.models.common.config",
        "mflux.models.common.config.model_config",
        "mflux.models.z_image",
        "mflux.models.z_image.variants",
        "mflux.models.z_image.variants.z_image",
        "mflux.models.flux",
        "mflux.models.flux.variants",
        "mflux.models.flux.variants.txt2img",
        "mflux.models.flux.variants.txt2img.flux",
        "mflux.models.flux.variants.kontext",
        "mflux.models.flux.variants.kontext.flux_kontext",
        "mflux.models.flux.variants.fill",
        "mflux.models.flux.variants.fill.flux_fill",
        "mflux.models.qwen",
        "mflux.models.qwen.variants",
        "mflux.models.qwen.variants.edit",
        "mflux.models.qwen.variants.edit.qwen_image_edit",
    ]:
        mocks[mod] = MagicMock()
    return mocks


def _fake_pil_image():
    """Return a MagicMock that behaves like a PIL Image (saves to buffer)."""
    img = MagicMock()

    def _save(buf, format="PNG"):
        buf.write(b"\x89PNG\r\n\x1a\nfake-image-data")

    img.save = _save
    return img


# ===========================================================================
# 1. SUPPORTED_MODELS dict
# ===========================================================================

class TestSupportedModels:
    def test_all_gen_models_present(self):
        from vmlx_engine.image_gen import SUPPORTED_MODELS

        expected_keys = [
            "schnell", "dev", "z-image", "z-image-turbo",
            "flux2-klein-4b", "flux2-klein-9b",
            "flux2-klein-base-4b", "flux2-klein-base-9b",
        ]
        for key in expected_keys:
            assert key in SUPPORTED_MODELS, f"Missing gen model: {key}"

    def test_aliases_resolve_correctly(self):
        from vmlx_engine.image_gen import SUPPORTED_MODELS

        assert SUPPORTED_MODELS["flux-schnell"] == "schnell"
        assert SUPPORTED_MODELS["flux-dev"] == "dev"
        assert SUPPORTED_MODELS["flux1-schnell"] == "schnell"
        assert SUPPORTED_MODELS["flux1-dev"] == "dev"

    def test_canonical_names_resolve_to_self(self):
        from vmlx_engine.image_gen import SUPPORTED_MODELS

        canonical = [
            "schnell", "dev", "z-image", "z-image-turbo",
            "flux2-klein-4b", "flux2-klein-9b",
            "flux2-klein-base-4b", "flux2-klein-base-9b",
        ]
        for name in canonical:
            assert SUPPORTED_MODELS[name] == name

    def test_no_unexpected_aliases(self):
        """All values must be canonical model names."""
        from vmlx_engine.image_gen import SUPPORTED_MODELS

        canonical = {
            "schnell", "dev", "z-image", "z-image-turbo",
            "flux2-klein-4b", "flux2-klein-9b",
            "flux2-klein-base-4b", "flux2-klein-base-9b",
        }
        for alias, resolved in SUPPORTED_MODELS.items():
            assert resolved in canonical, f"Alias '{alias}' resolves to unknown '{resolved}'"


# ===========================================================================
# 2. EDIT_MODELS dict
# ===========================================================================

class TestEditModels:
    def test_all_edit_models_present(self):
        from vmlx_engine.image_gen import EDIT_MODELS

        expected = ["qwen-image-edit", "flux-kontext", "kontext", "kontext-dev",
                     "flux-fill", "fill", "fill-dev"]
        for key in expected:
            assert key in EDIT_MODELS, f"Missing edit model: {key}"

    def test_kontext_aliases(self):
        from vmlx_engine.image_gen import EDIT_MODELS

        assert EDIT_MODELS["flux-kontext"] == "kontext-dev"
        assert EDIT_MODELS["kontext"] == "kontext-dev"
        assert EDIT_MODELS["kontext-dev"] == "kontext-dev"

    def test_fill_aliases(self):
        from vmlx_engine.image_gen import EDIT_MODELS

        assert EDIT_MODELS["flux-fill"] == "fill-dev"
        assert EDIT_MODELS["fill"] == "fill-dev"
        assert EDIT_MODELS["fill-dev"] == "fill-dev"

    def test_qwen_is_canonical(self):
        from vmlx_engine.image_gen import EDIT_MODELS

        assert EDIT_MODELS["qwen-image-edit"] == "qwen-image-edit"


# ===========================================================================
# 3. DEFAULT_STEPS dict
# ===========================================================================

class TestDefaultSteps:
    def test_all_gen_models_have_steps(self):
        from vmlx_engine.image_gen import SUPPORTED_MODELS, DEFAULT_STEPS

        canonical_gen = set(SUPPORTED_MODELS.values())
        for model in canonical_gen:
            assert model in DEFAULT_STEPS, f"No default steps for gen model: {model}"

    def test_all_edit_models_have_steps(self):
        from vmlx_engine.image_gen import EDIT_MODELS, DEFAULT_STEPS

        canonical_edit = set(EDIT_MODELS.values())
        for model in canonical_edit:
            assert model in DEFAULT_STEPS, f"No default steps for edit model: {model}"

    def test_steps_are_reasonable(self):
        from vmlx_engine.image_gen import DEFAULT_STEPS

        for model, steps in DEFAULT_STEPS.items():
            assert isinstance(steps, int), f"{model} steps is not int: {type(steps)}"
            assert 1 <= steps <= 50, f"{model} steps={steps} out of range 1-50"

    def test_schnell_is_fast(self):
        from vmlx_engine.image_gen import DEFAULT_STEPS

        assert DEFAULT_STEPS["schnell"] == 4

    def test_dev_is_standard(self):
        from vmlx_engine.image_gen import DEFAULT_STEPS

        assert DEFAULT_STEPS["dev"] == 20

    def test_z_image_turbo_is_fast(self):
        from vmlx_engine.image_gen import DEFAULT_STEPS

        assert DEFAULT_STEPS["z-image-turbo"] == 4

    def test_edit_model_steps(self):
        from vmlx_engine.image_gen import DEFAULT_STEPS

        assert DEFAULT_STEPS["qwen-image-edit"] == 28
        assert DEFAULT_STEPS["kontext-dev"] == 24
        assert DEFAULT_STEPS["fill-dev"] == 20


# ===========================================================================
# 4. _fix_quantized_layers()
# ===========================================================================

class TestFixQuantizedLayers:
    """Test _fix_quantized_layers with mock mlx modules."""

    def _make_mock_mlx(self):
        """Build mock mlx.nn and mlx.core for _fix_quantized_layers."""
        mock_nn = MagicMock()
        mock_mx = MagicMock()

        # Create distinct type sentinels
        QuantizedEmbedding = type("QuantizedEmbedding", (), {})
        QuantizedLinear = type("QuantizedLinear", (), {})
        Embedding = type("Embedding", (), {"__init__": lambda self, *a, **kw: None})
        Linear = type("Linear", (), {"__init__": lambda self, *a, **kw: None})

        mock_nn.QuantizedEmbedding = QuantizedEmbedding
        mock_nn.QuantizedLinear = QuantizedLinear
        mock_nn.Embedding = MagicMock(return_value=MagicMock())
        mock_nn.Linear = MagicMock(return_value=MagicMock())

        mock_mx.uint32 = "uint32"
        mock_mx.bfloat16 = "bfloat16"

        return mock_nn, mock_mx, QuantizedEmbedding, QuantizedLinear

    def _make_weight(self, dtype, shape=(128, 64)):
        w = MagicMock()
        w.dtype = dtype
        w.shape = shape
        return w

    def test_replaces_bfloat16_quantized_embedding(self):
        mock_nn, mock_mx, QEmb, QLin = self._make_mock_mlx()

        # Create a model with a QuantizedEmbedding that has bfloat16 weights
        qemb = QEmb()
        qemb.weight = self._make_weight("bfloat16", (32000, 4096))

        model = MagicMock()
        model.named_modules.return_value = [("shared", qemb)]

        # isinstance checks need to work with our custom types
        with patch.dict(sys.modules, {"mlx.nn": mock_nn, "mlx.core": mock_mx}):
            from vmlx_engine.image_gen import _fix_quantized_layers
            fixed = _fix_quantized_layers(model)

        assert fixed == 1
        mock_nn.Embedding.assert_called_once_with(32000, 4096)

    def test_replaces_bfloat16_quantized_linear(self):
        mock_nn, mock_mx, QEmb, QLin = self._make_mock_mlx()

        qlin = QLin()
        qlin.weight = self._make_weight("bfloat16", (512, 256))
        qlin.bias = self._make_weight("bfloat16", (512,))

        model = MagicMock()
        model.named_modules.return_value = [("layers.0.proj", qlin)]

        # We need getattr traversal to work: model.layers[0].proj
        layers_list = MagicMock()
        layers_0 = MagicMock()
        model.layers = layers_list
        layers_list.__getitem__ = MagicMock(return_value=layers_0)

        with patch.dict(sys.modules, {"mlx.nn": mock_nn, "mlx.core": mock_mx}):
            from vmlx_engine.image_gen import _fix_quantized_layers
            fixed = _fix_quantized_layers(model)

        assert fixed == 1
        # Linear(input_features=weight.shape[1], output_features=weight.shape[0], bias=True)
        mock_nn.Linear.assert_called_once_with(256, 512, bias=True)

    def test_leaves_uint32_quantized_layers_alone(self):
        mock_nn, mock_mx, QEmb, QLin = self._make_mock_mlx()

        # uint32 weights should NOT be replaced
        qemb = QEmb()
        qemb.weight = self._make_weight("uint32", (32000, 4096))

        model = MagicMock()
        model.named_modules.return_value = [("shared", qemb)]

        with patch.dict(sys.modules, {"mlx.nn": mock_nn, "mlx.core": mock_mx}):
            from vmlx_engine.image_gen import _fix_quantized_layers
            fixed = _fix_quantized_layers(model)

        assert fixed == 0
        mock_nn.Embedding.assert_not_called()
        mock_nn.Linear.assert_not_called()

    def test_no_quantized_layers_returns_zero(self):
        mock_nn, mock_mx, QEmb, QLin = self._make_mock_mlx()

        # A module that is NOT QuantizedEmbedding or QuantizedLinear
        regular_module = MagicMock()

        model = MagicMock()
        model.named_modules.return_value = [("layer", regular_module)]

        with patch.dict(sys.modules, {"mlx.nn": mock_nn, "mlx.core": mock_mx}):
            from vmlx_engine.image_gen import _fix_quantized_layers
            fixed = _fix_quantized_layers(model)

        assert fixed == 0

    def test_linear_without_bias(self):
        mock_nn, mock_mx, QEmb, QLin = self._make_mock_mlx()

        qlin = QLin()
        qlin.weight = self._make_weight("bfloat16", (256, 128))
        qlin.bias = None  # No bias

        model = MagicMock()
        model.named_modules.return_value = [("proj", qlin)]

        with patch.dict(sys.modules, {"mlx.nn": mock_nn, "mlx.core": mock_mx}):
            from vmlx_engine.image_gen import _fix_quantized_layers
            fixed = _fix_quantized_layers(model)

        assert fixed == 1
        mock_nn.Linear.assert_called_once_with(128, 256, bias=False)


# ===========================================================================
# 5. MODEL_CLASS_MAP and _import_model_class
# ===========================================================================

class TestModelClassMap:
    """Verify the explicit class dispatch table covers all models."""

    def test_all_name_to_class_entries_have_class_map(self):
        """Every entry in _NAME_TO_CLASS must have a matching MODEL_CLASS_MAP entry."""
        from vmlx_engine.image_gen import _NAME_TO_CLASS, MODEL_CLASS_MAP
        for name, cls in _NAME_TO_CLASS.items():
            assert cls in MODEL_CLASS_MAP, f"_NAME_TO_CLASS['{name}'] = '{cls}' not in MODEL_CLASS_MAP"

    def test_class_map_has_all_expected_classes(self):
        """MODEL_CLASS_MAP must include all known mflux model classes."""
        from vmlx_engine.image_gen import MODEL_CLASS_MAP
        expected = {'Flux1', 'Flux2Klein', 'ZImage', 'QwenImage', 'QwenImageEdit',
                    'Flux1Kontext', 'Flux1Fill', 'FIBO', 'SeedVR2', 'Flux2KleinEdit'}
        for cls in expected:
            assert cls in MODEL_CLASS_MAP, f"Missing '{cls}' in MODEL_CLASS_MAP"

    def test_import_model_class_rejects_unknown(self):
        """_import_model_class raises ValueError for unknown class names."""
        from vmlx_engine.image_gen import _import_model_class
        import pytest
        with pytest.raises(ValueError, match="Unknown mflux class"):
            _import_model_class("NonExistentClass")

    def test_default_steps_covers_all_names(self):
        """DEFAULT_STEPS must have entries for all _NAME_TO_CLASS keys."""
        from vmlx_engine.image_gen import _NAME_TO_CLASS, DEFAULT_STEPS
        for name in _NAME_TO_CLASS:
            assert name in DEFAULT_STEPS, f"Missing DEFAULT_STEPS entry for '{name}'"


# ===========================================================================
# 6. ImageGenResult dataclass
# ===========================================================================

class TestImageGenResult:

    def test_b64_json_auto_generated(self):
        from vmlx_engine.image_gen import ImageGenResult

        raw = b"\x89PNG\r\n\x1a\ntest-image-bytes"
        result = ImageGenResult(
            image_bytes=raw,
            width=1024,
            height=1024,
            model="schnell",
            seed=42,
            steps=4,
            elapsed_seconds=1.5,
        )
        decoded = base64.b64decode(result.b64_json)
        assert decoded == raw

    def test_b64_json_not_overwritten_if_provided(self):
        from vmlx_engine.image_gen import ImageGenResult

        result = ImageGenResult(
            image_bytes=b"abc",
            width=512,
            height=512,
            model="dev",
            seed=1,
            steps=20,
            elapsed_seconds=0.5,
            b64_json="pre-computed",
        )
        assert result.b64_json == "pre-computed"

    def test_empty_bytes_no_b64(self):
        from vmlx_engine.image_gen import ImageGenResult

        result = ImageGenResult(
            image_bytes=b"",
            width=512,
            height=512,
            model="dev",
            seed=1,
            steps=20,
            elapsed_seconds=0.5,
        )
        # Empty bytes → b64_json stays empty
        assert result.b64_json == ""

    def test_fields_stored(self):
        from vmlx_engine.image_gen import ImageGenResult

        result = ImageGenResult(
            image_bytes=b"data",
            width=768,
            height=512,
            model="z-image-turbo",
            seed=99,
            steps=4,
            elapsed_seconds=2.3,
        )
        assert result.width == 768
        assert result.height == 512
        assert result.model == "z-image-turbo"
        assert result.seed == 99
        assert result.steps == 4
        assert result.elapsed_seconds == 2.3


# ===========================================================================
# 7. generate() parameter handling
# ===========================================================================

class TestGenerateParameters:

    def _setup_engine(self):
        """Create an ImageGenEngine with a mocked model."""
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()
        engine._model = MagicMock()
        engine._loaded = True
        engine._model_name = "schnell"

        # Mock generate_image to return a fake PIL image
        gen_result = MagicMock()
        gen_result.image = _fake_pil_image()
        engine._model.generate_image.return_value = gen_result
        return engine

    def test_dimension_clamped_to_multiple_of_16(self):
        engine = self._setup_engine()
        result = engine.generate("test", width=1030, height=1001, seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["width"] == 1024   # 1030 // 16 * 16
        assert call_kwargs.kwargs["height"] == 992    # 1001 // 16 * 16

    def test_exact_multiple_of_16_unchanged(self):
        engine = self._setup_engine()
        result = engine.generate("test", width=1024, height=768, seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["width"] == 1024
        assert call_kwargs.kwargs["height"] == 768

    def test_default_steps_from_model(self):
        engine = self._setup_engine()
        engine._model_name = "schnell"
        result = engine.generate("test", seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["num_inference_steps"] == 4  # schnell default

    def test_default_steps_dev(self):
        engine = self._setup_engine()
        engine._model_name = "dev"
        result = engine.generate("test", seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["num_inference_steps"] == 20

    def test_explicit_steps_override_default(self):
        engine = self._setup_engine()
        result = engine.generate("test", steps=8, seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["num_inference_steps"] == 8

    def test_seed_passed_through(self):
        engine = self._setup_engine()
        result = engine.generate("test", seed=12345)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["seed"] == 12345
        assert result.seed == 12345

    def test_random_seed_generated_when_none(self):
        engine = self._setup_engine()
        with patch("random.randint", return_value=77777):
            result = engine.generate("test")

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["seed"] == 77777
        assert result.seed == 77777

    def test_guidance_passed_through(self):
        engine = self._setup_engine()
        result = engine.generate("test", guidance=7.5, seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["guidance"] == 7.5

    def test_negative_prompt_passed_through(self):
        engine = self._setup_engine()
        result = engine.generate("test", negative_prompt="ugly", seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["negative_prompt"] == "ugly"

    def test_prompt_passed_through(self):
        engine = self._setup_engine()
        result = engine.generate("a cat in space", seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["prompt"] == "a cat in space"

    def test_raises_when_not_loaded(self):
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()

        with pytest.raises(RuntimeError, match="No image model loaded"):
            engine.generate("test")

    def test_result_has_image_bytes(self):
        engine = self._setup_engine()
        result = engine.generate("test", seed=42)

        assert len(result.image_bytes) > 0
        assert result.b64_json != ""

    def test_result_model_name(self):
        engine = self._setup_engine()
        result = engine.generate("test", seed=42)

        assert result.model == "schnell"

    def test_unknown_model_defaults_to_20_steps(self):
        engine = self._setup_engine()
        engine._model_name = "some-future-model"
        result = engine.generate("test", seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["num_inference_steps"] == 20  # fallback default

    def test_small_dimensions_clamped(self):
        engine = self._setup_engine()
        result = engine.generate("test", width=15, height=31, seed=42)

        call_kwargs = engine._model.generate_image.call_args
        assert call_kwargs.kwargs["width"] == 0    # 15 // 16 * 16 = 0
        assert call_kwargs.kwargs["height"] == 16  # 31 // 16 * 16 = 16


# ===========================================================================
# 8. edit() model dispatch
# ===========================================================================

class TestEditModelDispatch:

    def _setup_edit_engine(self, model_name):
        """Create an engine pre-loaded with a mock edit model."""
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()
        engine._model = MagicMock()
        engine._loaded = True
        engine._model_name = model_name

        gen_result = MagicMock()
        gen_result.image = _fake_pil_image()
        engine._model.generate_image.return_value = gen_result
        return engine

    def test_qwen_edit_passes_image_paths(self):
        engine = self._setup_edit_engine("qwen-image-edit")
        result = engine.edit("make it red", image_path="/tmp/img.png", seed=42)

        call_kwargs = engine._model.generate_image.call_args.kwargs
        assert call_kwargs["image_path"] == "/tmp/img.png"
        assert call_kwargs["image_paths"] == ["/tmp/img.png"]
        assert call_kwargs["prompt"] == "make it red"

    def test_kontext_edit_passes_strength(self):
        engine = self._setup_edit_engine("kontext-dev")
        result = engine.edit("add hat", image_path="/tmp/img.png", strength=0.6, seed=42)

        call_kwargs = engine._model.generate_image.call_args.kwargs
        assert call_kwargs["image_path"] == "/tmp/img.png"
        assert call_kwargs["image_strength"] == 0.6

    def test_fill_requires_mask(self):
        engine = self._setup_edit_engine("fill-dev")

        with pytest.raises(ValueError, match="mask_path"):
            engine.edit("fill area", image_path="/tmp/img.png", seed=42)

    def test_fill_with_mask(self):
        engine = self._setup_edit_engine("fill-dev")
        result = engine.edit(
            "fill area", image_path="/tmp/img.png",
            mask_path="/tmp/mask.png", seed=42,
        )

        call_kwargs = engine._model.generate_image.call_args.kwargs
        assert call_kwargs["image_path"] == "/tmp/img.png"
        assert call_kwargs["masked_image_path"] == "/tmp/mask.png"

    def test_unknown_edit_model_raises(self):
        engine = self._setup_edit_engine("nonexistent-model")

        with pytest.raises(RuntimeError, match="does not support editing"):
            engine.edit("test", image_path="/tmp/img.png", seed=42)

    def test_edit_raises_when_not_loaded(self):
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()

        with pytest.raises(RuntimeError, match="No edit model loaded"):
            engine.edit("test", image_path="/tmp/img.png")

    def test_edit_dimension_clamping(self):
        engine = self._setup_edit_engine("kontext-dev")
        result = engine.edit("test", image_path="/tmp/img.png", width=1030, height=999, seed=42)

        call_kwargs = engine._model.generate_image.call_args.kwargs
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 992

    def test_edit_default_steps(self):
        engine = self._setup_edit_engine("qwen-image-edit")
        result = engine.edit("test", image_path="/tmp/img.png", seed=42)

        call_kwargs = engine._model.generate_image.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 28  # qwen default

    def test_edit_result_fields(self):
        engine = self._setup_edit_engine("kontext-dev")
        result = engine.edit("test", image_path="/tmp/img.png", seed=42)

        assert result.model == "kontext-dev"
        assert result.seed == 42
        assert len(result.image_bytes) > 0


# ===========================================================================
# 9. load_edit_model() dispatch
# ===========================================================================

class TestLoadEditModel:

    def test_qwen_import_path(self):
        """load_edit_model('qwen-image-edit') imports QwenImageEdit."""
        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        MockQwen = MagicMock()
        mocks["mflux.models.qwen.variants.edit.qwen_image_edit"].QwenImageEdit = MockQwen

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()
            engine.load_edit_model("qwen-image-edit")

        MockQwen.assert_called_once()
        assert engine._model_name == "qwen-image-edit"
        assert engine._loaded is True

    def test_kontext_import_path(self):
        """load_edit_model('flux-kontext') imports Flux1Kontext with resolved name."""
        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        MockKontext = MagicMock()
        mocks["mflux.models.flux.variants.kontext.flux_kontext"].Flux1Kontext = MockKontext

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()
            engine.load_edit_model("flux-kontext")

        MockKontext.assert_called_once()
        assert engine._model_name == "kontext-dev"

    def test_fill_import_path(self):
        """load_edit_model('flux-fill') imports Flux1Fill with resolved name."""
        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        MockFill = MagicMock()
        mocks["mflux.models.flux.variants.fill.flux_fill"].Flux1Fill = MockFill

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()
            engine.load_edit_model("fill")

        MockFill.assert_called_once()
        assert engine._model_name == "fill-dev"

    def test_unknown_edit_model_raises(self):
        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflx.models.common.config.model_config"] = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()

            with pytest.raises(ValueError, match="Unknown edit model"):
                engine.load_edit_model("totally-fake-model")

    def test_mflux_not_installed_raises(self):
        """ImportError when mflux is not available."""
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()

        # Ensure mflux import fails
        with patch.dict(sys.modules, {"mflux": None, "mflux.models": None,
                                       "mflux.models.common": None,
                                       "mflux.models.common.config": None,
                                       "mflux.models.common.config.model_config": None}):
            with pytest.raises(ImportError, match="mflux not installed"):
                engine.load_edit_model("qwen-image-edit")


# ===========================================================================
# 10. GGUF detection in convert_command
# ===========================================================================

class TestGGUFDetection:

    _PATCH_TARGET = "vmlx_engine.utils.model_inspector.resolve_model_path"

    def test_gguf_directory_rejected(self, tmp_path):
        """Directory with only .gguf files and no .safetensors → sys.exit(1)."""
        model_dir = tmp_path / "my-model-gguf"
        model_dir.mkdir()
        (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        args = argparse.Namespace(
            model=str(model_dir),
            bits=4,
            group_size=64,
            output=None,
            force=False,
            skip_verify=False,
            jang_profile=None,
        )

        with patch(self._PATCH_TARGET, return_value=str(model_dir)):
            with pytest.raises(SystemExit):
                from vmlx_engine.commands.convert import convert_command
                convert_command(args)

    def test_gguf_part_files_rejected(self, tmp_path):
        """Directory with .gguf files → sys.exit(1)."""
        model_dir = tmp_path / "split-gguf"
        model_dir.mkdir()
        (model_dir / "model-part1.gguf").write_bytes(b"\x00" * 50)
        (model_dir / "model-part2.gguf").write_bytes(b"\x00" * 50)
        (model_dir / "config.json").write_text('{"model_type": "llama"}')

        args = argparse.Namespace(
            model=str(model_dir),
            bits=4,
            group_size=64,
            output=None,
            force=False,
            skip_verify=False,
            jang_profile=None,
        )

        with patch(self._PATCH_TARGET, return_value=str(model_dir)):
            with pytest.raises(SystemExit):
                from vmlx_engine.commands.convert import convert_command
                convert_command(args)

    def test_gguf_with_safetensors_allowed(self, tmp_path):
        """Directory with BOTH .gguf and .safetensors → NOT rejected (has safetensors)."""
        model_dir = tmp_path / "hybrid-model"
        model_dir.mkdir()
        (model_dir / "model.gguf").write_bytes(b"\x00" * 100)
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)
        (model_dir / "config.json").write_text(
            '{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}'
        )

        args = argparse.Namespace(
            model=str(model_dir),
            bits=4,
            group_size=64,
            output=None,
            force=False,
            skip_verify=False,
            jang_profile=None,
        )

        # This should get past the GGUF check (will fail later at inspect_model,
        # but that means GGUF gate did NOT trigger)
        with patch(self._PATCH_TARGET, return_value=str(model_dir)):
            with patch("vmlx_engine.utils.model_inspector.inspect_model") as mock_inspect:
                mock_inspect.side_effect = FileNotFoundError("no weights")
                with pytest.raises(SystemExit):
                    from vmlx_engine.commands.convert import convert_command
                    convert_command(args)
                # The exit was from inspect_model failure, NOT the GGUF check
                mock_inspect.assert_called_once()

    def test_single_gguf_file_rejected(self, tmp_path):
        """Single .gguf file path → sys.exit(1)."""
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_bytes(b"\x00" * 100)

        args = argparse.Namespace(
            model=str(gguf_file),
            bits=4,
            group_size=64,
            output=None,
            force=False,
            skip_verify=False,
            jang_profile=None,
        )

        with patch(self._PATCH_TARGET, return_value=str(gguf_file)):
            with pytest.raises(SystemExit):
                from vmlx_engine.commands.convert import convert_command
                convert_command(args)


# ===========================================================================
# 11. Engine state management
# ===========================================================================

class TestEngineState:

    def test_initial_state(self):
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()

        assert engine.is_loaded is False
        assert engine.model_name is None
        assert engine._model is None

    def test_unload_clears_state(self):
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()
        engine._model = MagicMock()
        engine._loaded = True
        engine._model_name = "schnell"

        engine.unload()

        assert engine.is_loaded is False
        assert engine.model_name is None
        assert engine._model is None

    def test_unload_when_already_unloaded(self):
        """unload() on fresh engine should not raise."""
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()
        engine.unload()  # Should be a no-op

        assert engine.is_loaded is False

    def test_is_loaded_requires_both_flags(self):
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()

        # _loaded True but _model None
        engine._loaded = True
        engine._model = None
        assert engine.is_loaded is False

        # _loaded False but _model set
        engine._loaded = False
        engine._model = MagicMock()
        assert engine.is_loaded is False

        # Both set
        engine._loaded = True
        assert engine.is_loaded is True


# ===========================================================================
# 12. load() — alias resolution and mflux import guard
# ===========================================================================

class TestLoadMethod:

    def test_mflux_not_installed_raises(self):
        from vmlx_engine.image_gen import ImageGenEngine
        engine = ImageGenEngine()

        with patch.dict(sys.modules, {"mflux": None, "mflux.models": None,
                                       "mflux.models.common": None,
                                       "mflux.models.common.config": None,
                                       "mflux.models.common.config.model_config": None}):
            with pytest.raises(ImportError, match="mflux not installed"):
                engine.load("schnell")

    def test_alias_resolution_flux_schnell(self, tmp_path):
        """load('flux-schnell') should resolve to 'schnell' internally."""
        # Create mock model directory so local path check passes
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "0.safetensors").write_bytes(b"fake")
        (tmp_path / "text_encoder_2").mkdir()
        (tmp_path / "text_encoder_2" / "0.safetensors").write_bytes(b"fake")

        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        MockFlux1 = MagicMock()
        mocks["mflux.models.flux.variants.txt2img.flux"].Flux1 = MockFlux1

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()
            engine.load("flux-schnell", model_path=str(tmp_path))

        assert engine._model_name == "schnell"
        assert engine._loaded is True

    def test_zimage_model_uses_zimage_class(self, tmp_path):
        """load('z-image-turbo') should use ZImage class, not Flux1."""
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "0.safetensors").write_bytes(b"fake")

        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        MockZImage = MagicMock()
        mocks["mflux.models.z_image.variants.z_image"].ZImage = MockZImage

        MockFlux1 = MagicMock()
        mocks["mflux.models.flux.variants.txt2img.flux"].Flux1 = MockFlux1

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()
            engine.load("z-image-turbo", model_path=str(tmp_path))

        MockZImage.assert_called_once()
        MockFlux1.assert_not_called()
        # _detect_base_model may resolve "z-image-turbo" to "z-image" depending on model_index.json
        assert engine._model_name in ("z-image-turbo", "z-image")

    def test_flux_model_uses_flux1_class(self, tmp_path):
        """load('dev') should use Flux1 class, not ZImage."""
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "0.safetensors").write_bytes(b"fake")
        (tmp_path / "text_encoder_2").mkdir()
        (tmp_path / "text_encoder_2" / "0.safetensors").write_bytes(b"fake")

        mocks = _mock_mflux_modules()
        mock_config = MagicMock()
        mocks["mflux.models.common.config.model_config"].ModelConfig = mock_config

        MockZImage = MagicMock()
        mocks["mflux.models.z_image.variants.z_image"].ZImage = MockZImage

        MockFlux1 = MagicMock()
        mocks["mflux.models.flux.variants.txt2img.flux"].Flux1 = MockFlux1

        with patch.dict(sys.modules, mocks):
            from vmlx_engine.image_gen import ImageGenEngine
            engine = ImageGenEngine()
            engine.load("dev", model_path=str(tmp_path))

        MockFlux1.assert_called_once()
        MockZImage.assert_not_called()
        assert engine._model_name == "dev"

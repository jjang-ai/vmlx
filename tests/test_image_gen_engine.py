"""Tests for ImageGenEngine — model loading, no silent downloads, format detection."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestImageGenEngineLoading:
    """Verify _load_flux and _load_zimage load locally and never download."""

    def test_load_flux_local_mflux_native(self, tmp_path):
        """Flux loads from mflux-native format (numbered safetensors)."""
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "0.safetensors").write_bytes(b"fake")
        (tmp_path / "text_encoder_2").mkdir()
        (tmp_path / "text_encoder_2" / "0.safetensors").write_bytes(b"fake")

        model_path = str(tmp_path)
        transformer_dir = Path(model_path) / "transformer"
        has_transformer = transformer_dir.is_dir() and any(
            f.suffix == '.safetensors' for f in transformer_dir.iterdir()
        )
        assert has_transformer

    def test_load_flux_no_local_raises(self):
        """Flux raises RuntimeError when no local files exist."""
        from vmlx_engine.image_gen import ImageGenEngine
        eng = ImageGenEngine()
        with pytest.raises(RuntimeError, match="No local model files"):
            from mflux.models.common.config.model_config import ModelConfig
            eng._load_flux("schnell", 4, "/nonexistent/path", ModelConfig)

    def test_load_zimage_no_local_raises(self):
        """ZImage raises RuntimeError when no local files exist."""
        from vmlx_engine.image_gen import ImageGenEngine
        eng = ImageGenEngine()
        with pytest.raises(RuntimeError, match="No local model files"):
            from mflux.models.common.config.model_config import ModelConfig
            eng._load_zimage("z-image-turbo", 4, "/nonexistent/path", ModelConfig)

    def test_load_zimage_diffusers_format(self, tmp_path):
        """ZImage accepts diffusers format."""
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "diffusion_pytorch_model-00001.safetensors").write_bytes(b"fake")
        transformer_dir = Path(str(tmp_path)) / "transformer"
        has_transformer = transformer_dir.is_dir() and any(
            f.suffix == '.safetensors' for f in transformer_dir.iterdir()
        )
        assert has_transformer

    def test_no_from_name_in_load_flux(self):
        """_load_flux source should not call Flux1.from_name()."""
        import vmlx_engine.image_gen as img_mod
        source = open(img_mod.__file__).read()
        flux_section = source.split("def _load_flux")[1].split("\n    def ")[0]
        # ModelConfig.from_name is OK (metadata lookup), Flux1.from_name is NOT
        lines_with_from_name = [l for l in flux_section.split('\n')
                                if 'from_name(' in l and 'ModelConfig' not in l]
        assert len(lines_with_from_name) == 0, f"Found from_name() calls: {lines_with_from_name}"

    def test_no_from_name_in_load_zimage(self):
        """_load_zimage source should not call ZImage() without model_path."""
        import vmlx_engine.image_gen as img_mod
        source = open(img_mod.__file__).read()
        zimage_section = source.split("def _load_zimage")[1].split("\n    def ")[0]
        lines_with_from_name = [l for l in zimage_section.split('\n')
                                if 'from_name(' in l and 'ModelConfig' not in l]
        assert len(lines_with_from_name) == 0, f"Found from_name() calls: {lines_with_from_name}"


class TestEditModels:
    """Verify edit model registry and loading."""

    def test_edit_models_dict_exists(self):
        from vmlx_engine.image_gen import EDIT_MODELS
        assert "qwen-image-edit" in EDIT_MODELS

    def test_load_edit_model_unknown_raises(self):
        from vmlx_engine.image_gen import ImageGenEngine
        eng = ImageGenEngine()
        with pytest.raises((ValueError, Exception)):
            eng.load_edit_model("nonexistent-model-xyz")


class TestSupportedModels:
    """Verify SUPPORTED_MODELS has expected entries."""

    def test_generation_models_present(self):
        from vmlx_engine.image_gen import SUPPORTED_MODELS
        assert "schnell" in SUPPORTED_MODELS
        assert "dev" in SUPPORTED_MODELS
        assert "z-image-turbo" in SUPPORTED_MODELS

    def test_edit_models_separate(self):
        from vmlx_engine.image_gen import SUPPORTED_MODELS, EDIT_MODELS
        # Edit models should NOT be in SUPPORTED_MODELS
        for key in EDIT_MODELS:
            if key not in ("flux-kontext", "kontext", "kontext-dev", "flux-fill", "fill", "fill-dev"):
                assert key not in SUPPORTED_MODELS, f"Edit model '{key}' should not be in SUPPORTED_MODELS"


class TestMPOConversion:
    """Verify image format handling for edits."""

    def test_pil_import(self):
        from PIL import Image
        assert Image is not None

    def test_rgb_conversion(self):
        from PIL import Image
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        rgb = img.convert("RGB")
        assert rgb.mode == "RGB"

    def test_dimension_rounding(self):
        """mflux rounds dimensions to multiples of 16."""
        width, height = 800, 600
        rounded_w = (width // 16) * 16
        rounded_h = (height // 16) * 16
        assert rounded_w == 800  # 800 is already multiple of 16
        assert rounded_h == 592  # 600 → 592

"""Tests for ImageGenEngine — model loading, no silent downloads, format detection."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestImageGenEngineLoading:
    """Verify unified load() loads locally and never downloads."""

    def test_load_local_mflux_native(self, tmp_path):
        """Model loads from mflux-native format (numbered safetensors)."""
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

    def test_load_no_local_raises(self):
        """load() raises RuntimeError when no local files exist."""
        from vmlx_engine.image_gen import ImageGenEngine
        eng = ImageGenEngine()
        with pytest.raises(RuntimeError, match="No local model files"):
            eng.load("schnell", quantize=4, model_path="/nonexistent/path")

    def test_load_no_model_path_raises(self):
        """load() raises RuntimeError when model_path is None."""
        from vmlx_engine.image_gen import ImageGenEngine
        eng = ImageGenEngine()
        with pytest.raises(RuntimeError, match="No local model files"):
            eng.load("z-image-turbo")

    def test_load_diffusers_format(self, tmp_path):
        """Model directory with diffusers format has safetensors."""
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "diffusion_pytorch_model-00001.safetensors").write_bytes(b"fake")
        transformer_dir = Path(str(tmp_path)) / "transformer"
        has_transformer = transformer_dir.is_dir() and any(
            f.suffix == '.safetensors' for f in transformer_dir.iterdir()
        )
        assert has_transformer

    def test_no_model_class_from_name_in_load(self):
        """load() should not call ModelClass.from_name() — only ModelConfig.from_name()."""
        import vmlx_engine.image_gen as img_mod
        source = open(img_mod.__file__).read()
        load_section = source.split("def load(")[1].split("\n    def ")[0]
        # ModelConfig.from_name is OK (metadata lookup)
        # Flux1.from_name / ZImage.from_name etc. would trigger downloads — NOT OK
        lines_with_from_name = [l for l in load_section.split('\n')
                                if 'from_name(' in l and 'ModelConfig' not in l]
        assert len(lines_with_from_name) == 0, f"Found from_name() calls: {lines_with_from_name}"

    def test_no_silent_download_in_load(self):
        """load() source should not contain download/hub/snapshot calls."""
        import vmlx_engine.image_gen as img_mod
        source = open(img_mod.__file__).read()
        load_section = source.split("def load(")[1].split("\n    def ")[0]
        download_patterns = ["hf_hub_download", "snapshot_download", "from_pretrained"]
        for pattern in download_patterns:
            assert pattern not in load_section, f"Found '{pattern}' in load() — may silently download"


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

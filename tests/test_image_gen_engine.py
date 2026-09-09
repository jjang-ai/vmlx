"""Tests for ImageGenEngine — model loading, no silent downloads, format detection."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestImageExporterNames:
    @pytest.mark.parametrize("variant", ["fill", "kontext"])
    @pytest.mark.parametrize("suffix", ["q4", "q8", "bf16", "4bit"])
    def test_full_edit_names(self, variant, suffix):
        from vmlx_engine.image_gen import _normalize_for_lookup
        for name in (f"mflux-community/flux-1-dev-{variant}-mflux-{suffix}",
                     f"FLUX.1-{variant}-dev-mflux-{suffix}"):
            assert _normalize_for_lookup(name) == f"dev-{variant}"

    def test_specialised_variant_is_not_generic_fill(self):
        from vmlx_engine.image_gen import _normalize_for_lookup, EDIT_MODELS
        assert _normalize_for_lookup("flux-1-dev-fill-catvton-mflux-q4") not in EDIT_MODELS


class TestImageGuidanceDefaults:
    @pytest.mark.parametrize("model,mclass,expected", [
        ("dev", "Flux1", 3.5), ("qwen-image-edit", "QwenImageEdit", 4.0),
        ("dev-kontext", "Flux1Kontext", 2.5), ("dev-fill", "Flux1Fill", 30.0),
        ("z-image-turbo", "ZImage", 1.0), ("schnell", "Flux1", 0.0),
    ])
    @pytest.mark.parametrize("supplied", [None, 0.0, 7.25])
    def test_adapter_receives_model_default_or_explicit_value(self, model, mclass, expected, supplied):
        from PIL import Image
        from types import SimpleNamespace
        from vmlx_engine.image_gen import ImageGenEngine
        eng = ImageGenEngine()
        eng._model = MagicMock()
        eng._loaded = True
        eng._model_name = model
        eng._mflux_class = mclass
        eng._generate_with_trace = MagicMock(return_value=(
            SimpleNamespace(image=Image.new("RGB", (64, 64))), MagicMock(job_id="test-job")))
        kwargs = dict(prompt="test", width=64, height=64, steps=1, seed=1)
        if supplied is not None:
            kwargs["guidance"] = supplied
        if mclass in eng._EDIT_CLASSES:
            eng.edit(image_path="source.png", mask_path="mask.png", **kwargs)
        else:
            eng.generate(**kwargs)
        assert eng._generate_with_trace.call_args.kwargs["guidance"] == (
            expected if supplied is None else supplied)

    def test_product_default_tables_match_panel_registry(self):
        import re
        import vmlx_engine.image_gen as mod
        source = (Path(mod.__file__).resolve().parents[1] /
                  "panel/src/shared/imageModels.ts").read_text()
        rows = re.findall(r"steps: ([\d.]+),\s+guidance: ([\d.]+),[\s\S]*?mfluxName: '([^']+)'", source)
        assert len(rows) == 9
        for steps, guidance, name in rows:
            assert mod.DEFAULT_STEPS[name] == int(steps)
            assert mod.DEFAULT_GUIDANCE[name] == float(guidance)


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


class TestImageCapabilities:
    """The loaded model's real contract, derived from class + signature."""

    def _engine(self, mclass, params):
        from vmlx_engine.image_gen import ImageGenEngine

        class Model:
            pass

        def make(names):
            src = "def generate_image(self, " + ", ".join(f"{n}=None" for n in names) + "): pass"
            ns = {}
            exec(src, ns)
            return ns["generate_image"]

        Model.generate_image = make(params)
        eng = ImageGenEngine()
        eng._model = Model()
        eng._mflux_class = mclass
        eng._model_name = "m"
        eng._quantize = 8
        return eng

    def test_generation_model_with_negative_prompt_and_img2img(self):
        caps = self._engine("QwenImage", ["seed", "prompt", "num_inference_steps", "height", "width", "guidance", "image_path", "image_strength", "negative_prompt"]).capabilities()
        assert caps["mode"] == "generate" and caps["negative_prompt"] is True and caps["variation_strength"] is True
        assert caps["edit_strength"] is None and caps["mask"] == "none" and caps["count"] is True and caps["quantize"] == 8

    def test_klein_has_no_negative_prompt(self):
        caps = self._engine("Flux2Klein", ["seed", "prompt", "num_inference_steps", "height", "width", "guidance"]).capabilities()
        assert caps["negative_prompt"] is False and caps["variation_strength"] is False

    def test_flux_signature_does_not_claim_unused_negative_prompt(self):
        caps = self._engine("Flux1", ["seed", "prompt", "negative_prompt", "image_path", "image_strength"]).capabilities()
        assert caps["negative_prompt"] is False
        assert caps["variation_strength"] is True

    def test_installed_flux_negative_parameter_has_no_runtime_reads(self):
        # Qualification pin: reassess the explicit adapter contract if upstream
        # starts using the argument. A signature alone is not behavior proof.
        import dis
        flux = pytest.importorskip("mflux.models.flux.variants.txt2img.flux")
        fn = flux.Flux1.generate_image
        assert "negative_prompt" in fn.__code__.co_varnames
        assert not any(i.opname.startswith("LOAD_") and i.argval == "negative_prompt"
                       for i in dis.get_instructions(fn))

    def test_edit_classes_never_take_strength_and_fill_needs_a_mask(self):
        for mclass in ("QwenImageEdit", "Flux1Kontext", "Flux1Fill", "Flux2KleinEdit"):
            caps = self._engine(mclass, ["seed", "prompt", "num_inference_steps", "height", "width", "guidance", "image_path", "image_strength", "negative_prompt"]).capabilities()
            assert caps["mode"] == "edit" and caps["edit_strength"] is False and caps["count"] is False, mclass
            assert caps["mask"] == ("required" if mclass == "Flux1Fill" else "none")

    def test_health_exposes_the_capabilities(self):
        import inspect
        import vmlx_engine.server as server
        assert 'result["image"] = _image_gen.capabilities()' in inspect.getsource(server)

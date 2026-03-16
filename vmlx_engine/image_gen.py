# SPDX-License-Identifier: Apache-2.0
"""
Image Generation Engine — Text-to-image using mflux (MLX-native Flux).

Supports: Flux-dev, Flux-schnell, Z-Image-Turbo, Flux-Klein, and other
mflux-compatible models. Provides OpenAI-compatible /v1/images/generations API.

Usage:
    engine = ImageGenEngine()
    engine.load("schnell")  # or "dev", "z-image-turbo", "flux2-klein-4b"
    image = engine.generate("A cat in space", width=1024, height=1024, steps=4)
"""

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# mflux base model names (from mflux CLI --base-model choices)
SUPPORTED_MODELS = {
    "schnell": "schnell",
    "dev": "dev",
    "z-image": "z-image",
    "z-image-turbo": "z-image-turbo",
    "flux2-klein-4b": "flux2-klein-4b",
    "flux2-klein-9b": "flux2-klein-9b",
    "flux2-klein-base-4b": "flux2-klein-base-4b",
    "flux2-klein-base-9b": "flux2-klein-base-9b",
    # Aliases for convenience
    "flux-schnell": "schnell",
    "flux-dev": "dev",
    "flux1-schnell": "schnell",
    "flux1-dev": "dev",
}

# Default inference steps per model (schnell is fast, dev needs more)
DEFAULT_STEPS = {
    "schnell": 4,
    "dev": 20,
    "z-image": 20,
    "z-image-turbo": 4,
    "flux2-klein-4b": 20,
    "flux2-klein-9b": 20,
    "flux2-klein-base-4b": 20,
    "flux2-klein-base-9b": 20,
}


@dataclass
class ImageGenResult:
    """Result from image generation."""
    image_bytes: bytes  # PNG image data
    width: int
    height: int
    model: str
    seed: int
    steps: int
    elapsed_seconds: float
    b64_json: str = ""  # Base64-encoded PNG for API response

    def __post_init__(self):
        if not self.b64_json and self.image_bytes:
            self.b64_json = base64.b64encode(self.image_bytes).decode('utf-8')


class ImageGenEngine:
    """MLX-native image generation engine using mflux."""

    def __init__(self):
        self._model = None
        self._model_name: str | None = None
        self._quantize: int | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def load(
        self,
        model_name: str,
        quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        """Load a Flux model for image generation.

        Args:
            model_name: Model identifier (e.g., "schnell", "dev", "z-image-turbo")
            quantize: Quantization bits (3, 4, 5, 6, 8, or None for full precision)
            model_path: Optional path to local model weights
        """
        try:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config.model_config import ModelConfig
        except ImportError:
            raise ImportError(
                "mflux not installed. Install with: pip install mflux"
            )

        # Resolve model name alias
        resolved = SUPPORTED_MODELS.get(model_name.lower(), model_name)

        logger.info(f"Loading image model: {resolved} (quantize={quantize})")
        start = time.perf_counter()

        # Detect the base model name from directory structure
        base_name = self._detect_base_model(model_path, resolved) if model_path else resolved

        # Detect quantization level from local model if present
        if model_path and quantize is None:
            try:
                import json
                cfg_path = Path(model_path) / "config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text())
                    if "quantization_config" in cfg:
                        quantize = cfg["quantization_config"].get("bits")
                        logger.info(f"Detected quantization from config: {quantize}-bit")
            except Exception:
                pass

        # Use from_name — mflux handles downloading, caching, and architecture-specific loading.
        # Try quantized first, fall back to unquantized if dequantize fails (MLX version mismatch).
        logger.info(f"Loading via mflux: {base_name} (quantize={quantize})")
        try:
            self._model = Flux1.from_name(
                model_name=base_name,
                quantize=quantize,
            )
        except (ValueError, RuntimeError) as e:
            if "dequantize" in str(e) or "uint32" in str(e):
                if quantize:
                    logger.warning(
                        f"Quantized loading failed ({e}). "
                        f"Falling back to unquantized (needs more RAM). "
                        f"This is a known mflux/MLX version compatibility issue."
                    )
                    self._model = Flux1.from_name(
                        model_name=base_name,
                        quantize=None,
                    )
                    quantize = None
                else:
                    raise
            else:
                raise

        elapsed = time.perf_counter() - start
        self._model_name = resolved
        self._quantize = quantize
        self._loaded = True
        logger.info(f"Image model loaded in {elapsed:.1f}s: {resolved}")

    def unload(self) -> None:
        """Unload the current model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            self._model_name = None
            # Release Metal memory back to the system
            try:
                import mlx.core as mx
                if hasattr(mx, 'clear_memory_cache'):
                    mx.clear_memory_cache()
            except Exception:
                pass
            logger.info("Image model unloaded")

    @staticmethod
    def _detect_base_model(model_path: str, fallback_name: str) -> str:
        """Detect the mflux base model type from a local directory.

        Checks model_index.json for _class_name, then falls back to directory name matching.
        Returns a name that ModelConfig.from_name() recognizes.
        """
        import json
        model_dir = Path(model_path)

        # Check model_index.json for class name hints
        index_path = model_dir / "model_index.json"
        if index_path.exists():
            try:
                idx = json.loads(index_path.read_text())
                class_name = idx.get("_class_name", "").lower()
                if "zimage" in class_name or "z-image" in class_name:
                    return "z-image-turbo"  # Z-Image uses turbo config
            except Exception:
                pass

        # Check directory name against known model names
        dir_lower = model_dir.name.lower()
        for known in SUPPORTED_MODELS:
            if known.replace("-", "") in dir_lower.replace("-", ""):
                return SUPPORTED_MODELS[known]

        # Check if any parent directory matches
        for part in model_dir.parts:
            part_lower = part.lower()
            for known in SUPPORTED_MODELS:
                if known.replace("-", "") in part_lower.replace("-", ""):
                    return SUPPORTED_MODELS[known]

        # Fallback to the provided name (may work if it's a valid mflux name)
        return fallback_name

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance: float = 3.5,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageGenResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            width: Output image width (must be multiple of 16)
            height: Output image height (must be multiple of 16)
            steps: Number of denoising steps (None = model default)
            guidance: Classifier-free guidance scale (higher = more prompt adherence)
            seed: Random seed for reproducibility (None = random)
            negative_prompt: What to avoid in the image

        Returns:
            ImageGenResult with PNG image data
        """
        if not self.is_loaded:
            raise RuntimeError("No image model loaded. Call load() first.")

        # Resolve defaults
        if steps is None:
            steps = DEFAULT_STEPS.get(self._model_name, 20)
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)

        # Ensure dimensions are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        logger.info(
            f"Generating image: {width}x{height}, {steps} steps, "
            f"guidance={guidance}, seed={seed}"
        )
        start = time.perf_counter()

        # Generate using mflux
        generated_image = self._model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            negative_prompt=negative_prompt,
        )

        elapsed = time.perf_counter() - start

        # Convert to PNG bytes
        pil_image = generated_image.image
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        logger.info(
            f"Image generated in {elapsed:.1f}s: {width}x{height}, "
            f"{len(image_bytes) / 1024:.0f} KB"
        )

        return ImageGenResult(
            image_bytes=image_bytes,
            width=width,
            height=height,
            model=self._model_name or "unknown",
            seed=seed,
            steps=steps,
            elapsed_seconds=elapsed,
        )

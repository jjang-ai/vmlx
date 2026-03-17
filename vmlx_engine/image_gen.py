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
from dataclasses import dataclass
from pathlib import Path

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

# Image editing model names (separate from generation models)
EDIT_MODELS = {
    "qwen-image-edit": "qwen-image-edit",
    "flux-kontext": "kontext-dev",
    "kontext": "kontext-dev",
    "kontext-dev": "kontext-dev",
    "flux-fill": "fill-dev",
    "fill": "fill-dev",
    "fill-dev": "fill-dev",
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
    # Edit models
    "qwen-image-edit": 28,
    "kontext-dev": 24,
    "fill-dev": 20,
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


def _fix_quantized_layers(model) -> int:
    """Fix QuantizedEmbedding/QuantizedLinear layers with non-uint32 weights.

    mflux creates quantized layers for ALL modules when quantize=N,
    but some saved weights (e.g., T5 shared.weight, CLIP embeddings,
    transformer.x_embedder) are stored as bfloat16 — not uint32.
    mx.dequantize/quantized_matmul fails on these.
    This replaces them with standard Embedding/Linear layers.
    """
    import mlx.nn as nn
    import mlx.core as mx

    fixed = 0
    replacements = []
    for name, module in model.named_modules():
        is_qemb = isinstance(module, nn.QuantizedEmbedding)
        is_qlin = isinstance(module, nn.QuantizedLinear)
        if (is_qemb or is_qlin) and hasattr(module, 'weight') and module.weight.dtype != mx.uint32:
            replacements.append((name, module, 'embedding' if is_qemb else 'linear'))

    for name, module, layer_type in replacements:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            if hasattr(parent, p):
                parent = getattr(parent, p)
            elif isinstance(parent, dict) and p in parent:
                parent = parent[p]
            else:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)

        if layer_type == 'embedding':
            replacement = nn.Embedding(module.weight.shape[0], module.weight.shape[1])
            replacement.weight = module.weight
        else:
            has_bias = hasattr(module, 'bias') and module.bias is not None
            replacement = nn.Linear(module.weight.shape[1], module.weight.shape[0], bias=has_bias)
            replacement.weight = module.weight
            if has_bias:
                replacement.bias = module.bias

        attr = parts[-1]
        if isinstance(parent, dict):
            parent[attr] = replacement
        elif attr.isdigit():
            parent[int(attr)] = replacement
        else:
            setattr(parent, attr, replacement)
        fixed += 1

    return fixed


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
            from mflux.models.common.config.model_config import ModelConfig
        except ImportError:
            raise ImportError("mflux not installed. Install with: pip install mflux")

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

        # Z-Image models use ZImage class (single text encoder, different architecture)
        # Flux models (schnell, dev, klein) use Flux1 class (dual text encoder)
        is_zimage = base_name in ("z-image", "z-image-turbo")

        if is_zimage:
            self._load_zimage(base_name, quantize, model_path, ModelConfig)
        else:
            self._load_flux(base_name, quantize, model_path, ModelConfig)

        # Fix quantized embeddings that have non-uint32 weights (mflux bug)
        if quantize and self._model is not None:
            fixed = _fix_quantized_layers(self._model)
            if fixed:
                logger.info(f"Fixed {fixed} non-quantized embedding layers")

        elapsed = time.perf_counter() - start
        self._model_name = base_name  # Use canonical mflux name for DEFAULT_STEPS lookup
        self._quantize = quantize
        self._loaded = True
        logger.info(f"Image model loaded in {elapsed:.1f}s: {base_name}")

    def _load_zimage(self, base_name: str, quantize: int | None, model_path: str | None, ModelConfig) -> None:
        """Load a Z-Image model (single text encoder, ZImage class).
        ZImage handles both mflux-native and diffusers format from local paths.
        """
        from mflux.models.z_image.variants.z_image import ZImage

        model_config = ModelConfig.from_name(base_name)

        # ZImage can load from local path in both mflux-native AND diffusers format
        if model_path and Path(model_path).is_dir():
            transformer_dir = Path(model_path) / "transformer"
            has_transformer = transformer_dir.is_dir() and any(
                f.suffix == '.safetensors' for f in transformer_dir.iterdir()
            )
            if has_transformer:
                logger.info(f"Loading ZImage from local path: {model_path}")
                try:
                    self._model = ZImage(
                        model_config=model_config,
                        quantize=quantize,
                        model_path=model_path,
                        lora_paths=[],
                    )
                    return
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ZImage from {model_path}: {e}. "
                        f"Try re-downloading from the Image tab."
                    ) from e

        # No local path — NEVER silently download from HuggingFace
        raise RuntimeError(
            f"No local model files found for {base_name}. "
            f"Download the model first from the Image tab."
        )

    def _load_flux(self, base_name: str, quantize: int | None, model_path: str | None, ModelConfig) -> None:
        """Load a Flux model (dual text encoder, Flux1 class)."""
        from mflux.models.flux.variants.txt2img.flux import Flux1

        model_config = ModelConfig.from_name(base_name)

        # Try local path — Flux1 handles both mflux-native and diffusers format
        if model_path and Path(model_path).is_dir():
            transformer_dir = Path(model_path) / "transformer"
            has_transformer = transformer_dir.is_dir() and any(
                f.suffix == '.safetensors' for f in transformer_dir.iterdir()
            )
            if has_transformer:
                logger.info(f"Loading Flux from local path: {model_path}")
                try:
                    self._model = Flux1(
                        model_config=model_config,
                        quantize=quantize,
                        model_path=model_path,
                        lora_paths=[],
                    )
                    return
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load from {model_path}: {e}. "
                        f"Try re-downloading from the Image tab."
                    ) from e

        # No local path — NEVER silently download from HuggingFace
        raise RuntimeError(
            f"No local model files found for {base_name}. "
            f"Download the model first from the Image tab."
        )

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
                    # Distinguish z-image-turbo from z-image by checking directory
                    # name or scheduler config hints
                    dir_lower = model_dir.name.lower()
                    if "turbo" in dir_lower:
                        return "z-image-turbo"
                    # Default: check scheduler — turbo uses fewer default steps
                    scheduler_cfg = idx.get("scheduler", [None, {}])
                    if isinstance(scheduler_cfg, list) and len(scheduler_cfg) > 1:
                        sched_class = str(scheduler_cfg[0]).lower()
                        if "euler" in sched_class:
                            return "z-image"  # Non-turbo uses Euler scheduler
                    return "z-image-turbo"  # Default to turbo for ZImage class
            except Exception:
                pass

        # Check directory name against known model names (longest first to avoid
        # partial matches, e.g., "z-image" matching before "z-image-turbo")
        dir_lower = model_dir.name.lower()
        for known in sorted(SUPPORTED_MODELS, key=len, reverse=True):
            if known.replace("-", "") in dir_lower.replace("-", ""):
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
        image_path: str | None = None,
        image_strength: float | None = None,
    ) -> ImageGenResult:
        """Generate an image from a text prompt, optionally using a source image (img2img).

        Args:
            prompt: Text description of the desired image
            width: Output image width (must be multiple of 16)
            height: Output image height (must be multiple of 16)
            steps: Number of denoising steps (None = model default)
            guidance: Classifier-free guidance scale (higher = more prompt adherence)
            seed: Random seed for reproducibility (None = random)
            negative_prompt: What to avoid in the image
            image_path: Path to source image for img2img (None = txt2img from scratch)
            image_strength: How much to change the source (0.0=keep, 1.0=full regen). Only used with image_path.

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

        is_img2img = image_path is not None and image_strength is not None
        logger.info(
            f"{'img2img' if is_img2img else 'txt2img'}: {width}x{height}, {steps} steps, "
            f"guidance={guidance}, seed={seed}"
            + (f", strength={image_strength}" if is_img2img else "")
        )
        start = time.perf_counter()

        # Generate using mflux (img2img when image_path provided, txt2img otherwise)
        kwargs: dict = dict(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            negative_prompt=negative_prompt,
        )
        if is_img2img:
            kwargs["image_path"] = image_path
            kwargs["image_strength"] = image_strength

        generated_image = self._model.generate_image(**kwargs)

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

    def load_edit_model(
        self,
        model_name: str,
        quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        """Load an image editing model.

        Args:
            model_name: Edit model identifier (e.g., "qwen-image-edit", "flux-kontext")
            quantize: Quantization bits (4, 8, or None for full precision)
            model_path: Optional path to local model weights
        """
        try:
            from mflux.models.common.config.model_config import ModelConfig
        except ImportError:
            raise ImportError("mflux not installed. Install with: pip install mflux")

        resolved = EDIT_MODELS.get(model_name.lower(), model_name)

        # Detect quantization level from local model config if not explicitly set
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

        logger.info(f"Loading edit model: {resolved} (quantize={quantize})")
        start = time.perf_counter()

        model_config = ModelConfig.from_name(resolved)

        if resolved == "qwen-image-edit":
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
            self._model = QwenImageEdit(
                model_config=model_config,
                quantize=quantize,
                model_path=model_path,
                lora_paths=[],
            )
        elif resolved == "kontext-dev":
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
            self._model = Flux1Kontext(
                model_config=model_config,
                quantize=quantize,
                model_path=model_path,
                lora_paths=[],
            )
        elif resolved == "fill-dev":
            from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
            self._model = Flux1Fill(
                model_config=model_config,
                quantize=quantize,
                model_path=model_path,
                lora_paths=[],
            )
        else:
            raise ValueError(f"Unknown edit model: {model_name}. Available: {list(EDIT_MODELS.keys())}")

        # Fix quantized embeddings that have non-uint32 weights (mflux bug)
        if quantize and self._model is not None:
            fixed = _fix_quantized_layers(self._model)
            if fixed:
                logger.info(f"Fixed {fixed} non-quantized embedding layers")

        elapsed = time.perf_counter() - start
        self._model_name = resolved
        self._quantize = quantize
        self._loaded = True
        logger.info(f"Edit model loaded in {elapsed:.1f}s: {resolved}")

    def edit(
        self,
        prompt: str,
        image_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance: float = 3.5,
        seed: int | None = None,
        strength: float = 0.75,
        negative_prompt: str | None = None,
        mask_path: str | None = None,
    ) -> ImageGenResult:
        """Edit an image using a loaded editing model.

        Args:
            prompt: Text instruction for the edit
            image_path: Path to the source image
            width: Output width
            height: Output height
            steps: Denoising steps (None = model default)
            guidance: Guidance scale
            seed: Random seed
            strength: How much to change the image (0.0 = no change, 1.0 = full regen)
            negative_prompt: What to avoid
            mask_path: Path to mask image (for inpainting with Flux Fill)

        Returns:
            ImageGenResult with edited PNG image data
        """
        if not self.is_loaded:
            raise RuntimeError("No edit model loaded. Call load_edit_model() first.")

        if steps is None:
            steps = DEFAULT_STEPS.get(self._model_name, 20)
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)

        width = (width // 16) * 16
        height = (height // 16) * 16

        logger.info(
            f"Editing image: model={self._model_name}, {width}x{height}, "
            f"{steps} steps, guidance={guidance}, strength={strength}, seed={seed}"
        )
        start = time.perf_counter()

        model_name = self._model_name or ""

        if model_name == "qwen-image-edit":
            generated_image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                image_path=image_path,
                image_paths=[image_path],
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                negative_prompt=negative_prompt,
            )
        elif model_name == "kontext-dev":
            generated_image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                image_path=image_path,
                image_strength=strength,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
        elif model_name == "fill-dev":
            if not mask_path:
                raise ValueError("Flux Fill requires a mask_path for inpainting")
            generated_image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                image_path=image_path,
                masked_image_path=mask_path,
                image_strength=strength,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
        else:
            raise RuntimeError(f"Model {model_name} does not support editing")

        elapsed = time.perf_counter() - start

        pil_image = generated_image.image
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        logger.info(
            f"Image edited in {elapsed:.1f}s: {width}x{height}, "
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

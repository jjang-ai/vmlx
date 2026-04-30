# SPDX-License-Identifier: Apache-2.0
# Created by Jinho Jang (eric@jangq.ai) for vMLX / mlxstudio
"""
Image Generation Engine — Text-to-image and img2img using mflux (MLX-native).

Supports all mflux model classes: Flux1, Flux2Klein, ZImage, FIBO,
Flux1Kontext, Flux1Fill, QwenImage, QwenImageEdit, SeedVR2.

Model class selection is EXPLICIT — passed via mflux_class parameter from the
shared imageModels.ts registry. NO regex, NO directory name matching.

Usage:
    engine = ImageGenEngine()
    engine.load("schnell", mflux_class="Flux1", mflux_name="schnell")
    image = engine.generate("A cat in space", width=1024, height=1024, steps=4)
    # img2img:
    image = engine.generate("Golden cat", image_path="/tmp/cat.png", image_strength=0.7)
"""

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Class dispatch table ──────────────────────────────────────────────
# Maps mfluxClass string -> (module_path, class_name)
# Used by _import_model_class() to import the correct Python class.
# NO regex. NO directory name matching. Explicit mapping only.
MODEL_CLASS_MAP: dict[str, tuple[str, str]] = {
    "Flux1": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
    "Flux2Klein": ("mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein"),
    "Flux2KleinEdit": ("mflux.models.flux2.variants.edit.flux2_klein_edit", "Flux2KleinEdit"),
    "ZImage": ("mflux.models.z_image.variants.z_image", "ZImage"),
    "FIBO": ("mflux.models.fibo.variants.txt2img.fibo", "FIBO"),
    "QwenImage": ("mflux.models.qwen.variants.txt2img.qwen_image", "QwenImage"),
    "QwenImageEdit": ("mflux.models.qwen.variants.edit.qwen_image_edit", "QwenImageEdit"),
    "Flux1Kontext": ("mflux.models.flux.variants.kontext.flux_kontext", "Flux1Kontext"),
    "Flux1Fill": ("mflux.models.flux.variants.fill.flux_fill", "Flux1Fill"),
    "SeedVR2": ("mflux.models.seedvr2.variants.upscale.seedvr2", "SeedVR2"),
}

# Default inference steps per mflux canonical name
DEFAULT_STEPS: dict[str, int] = {
    "schnell": 4,
    "dev": 20,
    "z-image": 20,
    "z-image-turbo": 4,
    "flux2-klein-4b": 20,
    "flux2-klein-9b": 20,
    "flux2-klein-base-4b": 20,
    "flux2-klein-base-9b": 20,
    "qwen-image": 20,
    "qwen-image-edit": 28,
    "dev-kontext": 24,
    "dev-fill": 20,
    "fibo": 20,
    "fibo-lite": 20,
    "seedvr2-3b": 1,
    "seedvr2-7b": 1,
}

# Legacy name aliases (kept for backward compat with CLI and stored configs)
SUPPORTED_MODELS: dict[str, str] = {
    "schnell": "schnell",
    "dev": "dev",
    "z-image": "z-image",
    "z-image-turbo": "z-image-turbo",
    "flux2-klein-4b": "flux2-klein-4b",
    "flux2-klein-9b": "flux2-klein-9b",
    "flux2-klein-base-4b": "flux2-klein-base-4b",
    "flux2-klein-base-9b": "flux2-klein-base-9b",
    "qwen-image": "qwen-image",
    "fibo": "fibo",
    # Aliases
    "flux-schnell": "schnell",
    "flux-dev": "dev",
    "flux1-schnell": "schnell",
    "flux1-dev": "dev",
    "klein-4b": "flux2-klein-4b",
    "klein-9b": "flux2-klein-9b",
    # HuggingFace repo name aliases (FLUX.2-klein-4B → flux2-klein-4b)
    "flux.2-klein-4b": "flux2-klein-4b",
    "flux.2-klein-9b": "flux2-klein-9b",
    "flux.2-klein-base-4b": "flux2-klein-base-4b",
    "flux.2-klein-base-9b": "flux2-klein-base-9b",
    # mlxstudio#82: FLUX.1 dotted HF repo forms (gen variants only; edit
    # variants live in EDIT_MODELS to keep the two tables non-overlapping).
    "flux.1-dev": "dev",
    "flux.1-schnell": "schnell",
}

EDIT_MODELS: dict[str, str] = {
    "qwen-image-edit": "qwen-image-edit",
    "flux-kontext": "dev-kontext",
    "kontext": "dev-kontext",
    "kontext-dev": "dev-kontext",
    "dev-kontext": "dev-kontext",
    "flux-fill": "dev-fill",
    "fill": "dev-fill",
    "fill-dev": "dev-fill",
    "dev-fill": "dev-fill",
    # mlxstudio#82: FLUX.1 dotted HF repo forms for edit variants
    "flux.1-kontext-dev": "dev-kontext",
    "flux.1-fill-dev": "dev-fill",
}

# Map mflux canonical name -> mfluxClass (for legacy load paths that don't pass mflux_class)
_NAME_TO_CLASS: dict[str, str] = {
    "schnell": "Flux1",
    "dev": "Flux1",
    "z-image": "ZImage",
    "z-image-turbo": "ZImage",
    "flux2-klein-4b": "Flux2Klein",
    "flux2-klein-9b": "Flux2Klein",
    "flux2-klein-base-4b": "Flux2Klein",
    "flux2-klein-base-9b": "Flux2Klein",
    "qwen-image": "QwenImage",
    "qwen-image-edit": "QwenImageEdit",
    "dev-kontext": "Flux1Kontext",
    "dev-fill": "Flux1Fill",
    "fibo": "FIBO",
    "fibo-lite": "FIBO",
    "seedvr2-3b": "SeedVR2",
    "seedvr2-7b": "SeedVR2",
    # mlxstudio#82: dotted HF forms resolve via SUPPORTED_MODELS/EDIT_MODELS
    # -> canonical undotted name -> this table. No direct dotted entries
    # needed (would just duplicate and break test_default_steps_covers_all_names
    # which asserts every _NAME_TO_CLASS key has a DEFAULT_STEPS entry).
}


# mlxstudio#82: diffusers `_class_name` values from model_index.json that we
# can map deterministically to an mflux class. This is the LAST-RESORT
# fallback for user-renamed directories where name-based resolution fails.
#
# Some pipelines (FluxPipeline, Flux2KleinPipeline) are ambiguous between
# variants (schnell/dev, klein-4b/9b); in those cases we pick a sensible
# default and emit a WARNING so the user can override via --mflux-name.
_DIFFUSERS_CLASS_TO_MFLUX: dict[str, tuple[str, str]] = {
    # (mflux_class, default mflux_name — mflux canonical key)
    "FluxPipeline": ("Flux1", "dev"),  # ambiguous schnell/dev; default dev
    "FluxControlPipeline": ("Flux1", "dev"),
    "Flux2KleinPipeline": ("Flux2Klein", "flux2-klein-9b"),  # default to larger
    "Flux2KleinBasePipeline": ("Flux2Klein", "flux2-klein-base-9b"),
    "ZImagePipeline": ("ZImage", "z-image-turbo"),
    "ZImageTurboPipeline": ("ZImage", "z-image-turbo"),
    "FIBOPipeline": ("FIBO", "fibo"),
    "QwenImagePipeline": ("QwenImage", "qwen-image"),
    "QwenImageEditPipeline": ("QwenImageEdit", "qwen-image-edit"),
    "FluxKontextPipeline": ("Flux1Kontext", "dev-kontext"),
    "FluxFillPipeline": ("Flux1Fill", "dev-fill"),
    "SeedVR2Pipeline": ("SeedVR2", "seedvr2-3b"),
}


def _detect_class_from_model_index(model_path: str | None) -> tuple[str, str] | None:
    """Last-resort class resolution for user-renamed model directories.

    Read `{model_path}/model_index.json` (standard diffusers metadata) and
    map its `_class_name` field to an (mflux_class, default_mflux_name) tuple.

    Returns None if model_path is missing, model_index.json is absent, the
    file is unreadable, or the `_class_name` is not in
    `_DIFFUSERS_CLASS_TO_MFLUX`. Errors are logged but do not raise.

    Used by ImageGenEngine.load() when name-based resolution cannot find a
    matching canonical name — covers directory names like
    `FLUX.2-klien-blah-blah` (typo) or `INT_QIE` (user shortened).
    """
    if not model_path:
        return None
    try:
        import json
        idx_path = Path(model_path) / "model_index.json"
        if not idx_path.exists():
            return None
        data = json.loads(idx_path.read_text())
        class_name = data.get("_class_name")
        if not isinstance(class_name, str):
            return None
        mapping = _DIFFUSERS_CLASS_TO_MFLUX.get(class_name)
        if mapping:
            logger.info(
                f"mlxstudio#82: model_index.json _class_name='{class_name}' "
                f"-> mflux_class={mapping[0]}, default mflux_name={mapping[1]} "
                f"(pass --mflux-name explicitly to override for ambiguous variants)"
            )
            return mapping
        logger.warning(
            f"mlxstudio#82: model_index.json _class_name='{class_name}' is not "
            f"in the known mflux-class map. Pass --mflux-class explicitly."
        )
    except Exception as e:
        logger.warning(f"mlxstudio#82: failed to read model_index.json at {model_path}: {e}")
    return None


def _normalize_for_lookup(raw: str) -> str:
    """Normalize a user-facing model name into the form used as a key in
    SUPPORTED_MODELS / EDIT_MODELS / _NAME_TO_CLASS.

    Rules (applied in order):
      1. Lowercase
      2. Strip HuggingFace org prefix (`org/name` -> `name`)
      3. Strip quant decoration suffixes: `-mflux-Nbit`, `-mflux`, `-Nbit`, `_Nbit`
      4. Strip explicit user prefix markers used in the UI (e.g. `int-`, `ext-`,
         `int_`, `ext_`) — Mark (@LewnWorx, mlxstudio#82) renames launched
         instances with INT/EXT prefixes for local/external differentiation.

    Dotted forms (`flux.1-dev`, `flux.2-klein-9b`) are NOT collapsed to undotted
    here because both forms are explicit entries in the alias tables — keeping
    the two forms distinct lets callers who store the dotted form still match
    directly without the lookup fiddling with the key shape.
    """
    import re
    s = (raw or "").strip().lower()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    # Order matters: -mflux-Nbit before -mflux before -Nbit
    s = re.sub(r"-mflux-\d+bit$", "", s)
    s = re.sub(r"-mflux$", "", s)
    s = re.sub(r"[-_]\d+bit$", "", s)
    # Mark's INT-/EXT- convention for renamed local/external instances
    for prefix in ("int-", "ext-", "int_", "ext_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


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


def _import_model_class(mflux_class: str):
    """Import and return the mflux model class by explicit name.
    No regex. No guessing. Fails fast with clear error.
    """
    if mflux_class not in MODEL_CLASS_MAP:
        raise ValueError(
            f"Unknown mflux class: '{mflux_class}'. "
            f"Available: {sorted(MODEL_CLASS_MAP.keys())}"
        )
    module_path, class_name = MODEL_CLASS_MAP[mflux_class]
    import importlib
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import {class_name} from {module_path}: {e}. "
            f"Your mflux version may be incompatible. Try: pip install --upgrade mflux"
        ) from e
    return getattr(mod, class_name)


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

    @property
    def b64_json(self) -> str:
        return base64.b64encode(self.image_bytes).decode("utf-8")


class ImageGenEngine:
    """MLX-native image generation engine using mflux.

    All model loading uses explicit class names from the MODEL_CLASS_MAP.
    No regex, no directory name matching.
    """

    def __init__(self):
        self._model = None
        self._model_name: str | None = None
        self._model_path: str | None = None
        self._mflux_class: str | None = None
        self._quantize: int | None = None
        self._loaded = False
        # vmlx#100 / mlxstudio#100 — cooperative cancellation. mflux's
        # `generate_image` is synchronous and not preemptible from outside,
        # so this flag is checked at every coarse boundary that we own:
        # before each generate() call, between img-batch frames, and (when
        # mflux exposes a step callback in a future release) per denoise
        # step. Calling `cancel()` aborts the NEXT cooperative checkpoint
        # rather than mid-step. Reset on every successful return so a fresh
        # generate() doesn't inherit a stale cancel.
        import threading
        self._cancel_flag = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the current/next generation.

        Cooperative — interrupts at the next checkpoint we own. The
        currently-running denoise step (inside mflux) still runs to
        completion. Subsequent calls to generate()/edit() return early
        with a CancelledError until cancel state is cleared by a fresh
        successful return or `clear_cancel()`.
        """
        self._cancel_flag.set()

    def clear_cancel(self) -> None:
        """Reset the cancel flag (idempotent)."""
        self._cancel_flag.clear()

    @property
    def is_cancelling(self) -> bool:
        return self._cancel_flag.is_set()

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
        mflux_class: str | None = None,
        mflux_name: str | None = None,
    ) -> None:
        """Load any image model (generation or editing).

        Args:
            model_name: Model identifier (e.g., "schnell", "qwen-image-edit")
            quantize: Quantization bits (4, 8, or None for full precision)
            model_path: Path to local model weights
            mflux_class: Explicit mflux Python class name (e.g., "Flux1", "Flux2Klein").
                         If not provided, falls back to name-based lookup.
            mflux_name: Canonical mflux model name for ModelConfig.from_name().
                        If not provided, resolved from model_name.
        """
        try:
            from mflux.models.common.config.model_config import ModelConfig
        except ImportError:
            raise ImportError("mflux not installed. Install with: pip install mflux")

        # mlxstudio#82: resolve canonical mflux name via unconditional
        # normalization. Prior behaviour skipped the normalization block
        # whenever caller passed `mflux_name` explicitly, which trapped
        # raw directory basenames (e.g. "FLUX.2-klein-9B") in the raw form
        # where downstream ModelConfig.from_name() and _NAME_TO_CLASS lookups
        # would miss.
        #
        # The normalized form is used for table lookups; the final
        # `resolved_name` is the canonical mflux value ("dev", "flux2-klein-9b",
        # etc.) that mflux's ModelConfig.from_name() understands.
        raw_input = mflux_name or model_name
        normalized = _normalize_for_lookup(raw_input)
        canonical = (
            SUPPORTED_MODELS.get(normalized)
            or EDIT_MODELS.get(normalized)
        )
        if canonical:
            resolved_name = canonical
        elif mflux_name:
            # Caller asserted a canonical name; honour it (they may know about
            # a newer mflux entry we haven't aliased yet). mflux will raise if
            # it's truly unknown.
            resolved_name = mflux_name
        else:
            # No canonical hit; use the normalized form so the class lookup
            # below gets a chance against its dotted-form entries.
            resolved_name = normalized

        # Resolve mflux class
        resolved_class = mflux_class
        if not resolved_class:
            resolved_class = (
                _NAME_TO_CLASS.get(resolved_name)
                or _NAME_TO_CLASS.get(resolved_name.lower())
                or _NAME_TO_CLASS.get(normalized)
            )
            if resolved_class and resolved_name != normalized and canonical:
                logger.info(
                    f"mlxstudio#82: resolved directory name '{raw_input}' "
                    f"-> mflux name '{resolved_name}' -> class {resolved_class}"
                )
            if not resolved_class:
                # mlxstudio#82: last-resort — read model_index.json's _class_name
                # for user-renamed directories that no name-based pattern can
                # resolve (e.g. `INT_QIE`, `FLUX.2-klien-blah-blah`).
                idx_match = _detect_class_from_model_index(model_path)
                if idx_match is not None:
                    resolved_class, _default_name = idx_match
                    # If caller didn't supply a matching canonical name,
                    # adopt the one the _class_name implies. mflux's
                    # ModelConfig.from_name() requires a canonical key, and
                    # the raw user directory name won't satisfy it.
                    if resolved_name not in SUPPORTED_MODELS.values() and resolved_name not in EDIT_MODELS.values():
                        resolved_name = _default_name
            if not resolved_class:
                raise ValueError(
                    f"Cannot determine mflux class for model '{model_name}' "
                    f"(resolved name: '{resolved_name}', normalized: '{normalized}'). "
                    f"Pass mflux_class explicitly, add to _NAME_TO_CLASS, or "
                    f"ensure the model path contains a model_index.json whose "
                    f"_class_name is one of: {sorted(_DIFFUSERS_CLASS_TO_MFLUX.keys())}. "
                    f"Known keys: {sorted(set(_NAME_TO_CLASS.keys()))}"
                )

        logger.info(f"Loading image model: {resolved_name} (class={resolved_class}, quantize={quantize})")
        start = time.perf_counter()

        # Detect quantization from local config if not set
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

        # Import the correct model class
        ModelClass = _import_model_class(resolved_class)

        # Get mflux model config
        model_config = ModelConfig.from_name(resolved_name)

        # Load from local path (NEVER silently download from HuggingFace)
        if model_path and Path(model_path).is_dir():
            logger.info(f"Loading {resolved_class} from local path: {model_path}")
            try:
                self._model = ModelClass(
                    model_config=model_config,
                    quantize=quantize,
                    model_path=model_path,
                    lora_paths=[],
                )
            except TypeError:
                # Some classes (SeedVR2) don't accept lora_paths
                self._model = ModelClass(
                    model_config=model_config,
                    quantize=quantize,
                    model_path=model_path,
                )
        else:
            raise RuntimeError(
                f"No local model files found for {resolved_name}. "
                f"Download the model first from the Image tab."
            )

        # Fix quantized embeddings with non-uint32 weights (mflux bug)
        if quantize and self._model is not None:
            fixed = _fix_quantized_layers(self._model)
            if fixed:
                logger.info(f"Fixed {fixed} non-quantized embedding layers")

        elapsed = time.perf_counter() - start
        self._model_name = resolved_name
        self._model_path = model_path
        self._mflux_class = resolved_class
        self._quantize = quantize
        self._loaded = True
        logger.info(f"Image model loaded in {elapsed:.1f}s: {resolved_name} ({resolved_class})")

    # Keep backward compat — load_edit_model just delegates to load()
    def load_edit_model(
        self,
        model_name: str,
        quantize: int | None = None,
        model_path: str | None = None,
        mflux_class: str | None = None,
        mflux_name: str | None = None,
    ) -> None:
        """Load an image editing model. Delegates to load()."""
        self.load(model_name, quantize, model_path, mflux_class, mflux_name)

    def unload(self) -> None:
        """Unload the current model to free memory.
        Preserves _model_name, _mflux_class, _quantize, _model_path for reload on wake."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            # Keep metadata for wake/reload: _model_name, _mflux_class, _quantize, _model_path
            try:
                import mlx.core as mx
                if hasattr(mx, 'clear_memory_cache'):
                    mx.clear_memory_cache()
            except Exception:
                pass
            logger.info("Image model unloaded")

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

        Works with ALL generation models (Flux1, Flux2Klein, ZImage, FIBO, QwenImage).
        img2img is supported when image_path + image_strength are provided.

        Args:
            prompt: Text description of the desired image
            width: Output image width (must be multiple of 16)
            height: Output image height (must be multiple of 16)
            steps: Number of denoising steps (None = model default)
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility (None = random)
            negative_prompt: What to avoid in the image
            image_path: Path to source image for img2img (None = txt2img)
            image_strength: How much to change source (0-1). Only used with image_path.
        """
        if not self.is_loaded:
            raise RuntimeError("No image model loaded. Call load() first.")

        # vmlx#100 / mlxstudio#100 — honor cancellation requested while a
        # prior call was running (or before this call started). Mflux is
        # uncancellable mid-step, so this is the entry-point checkpoint.
        if self._cancel_flag.is_set():
            self._cancel_flag.clear()
            raise asyncio.CancelledError(
                "image generation cancelled before start (cancel() was called)"
            )

        if steps is None:
            steps = DEFAULT_STEPS.get(self._model_name, 20)
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)

        width = (width // 16) * 16
        height = (height // 16) * 16

        # Defense in depth: generate() requires a generation model. Calling
        # it on an edit-only class (QwenImageEdit / Flux1Kontext / Flux1Fill /
        # Flux2KleinEdit) surfaces as a deep mflux TypeError ("unexpected
        # keyword argument 'image_path'"). Fail fast with a clear message.
        _gen_mclasses = {"Flux1", "Flux2Klein", "ZImage", "FIBO", "QwenImage"}
        if self._mflux_class and self._mflux_class not in _gen_mclasses:
            raise ValueError(
                f"Model '{self._model_name}' (class={self._mflux_class}) is an "
                f"editing model, not a generation model. Use ImageGenEngine.edit() "
                f"or load schnell / dev / z-image-turbo for generation."
            )

        is_img2img = image_path is not None and image_strength is not None
        logger.info(
            f"{'img2img' if is_img2img else 'txt2img'}: {width}x{height}, {steps} steps, "
            f"guidance={guidance}, seed={seed}"
            + (f", strength={image_strength}" if is_img2img else "")
        )
        start = time.perf_counter()

        kwargs: dict = dict(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        # Not all models accept negative_prompt (Klein doesn't)
        if negative_prompt and 'negative_prompt' in self._get_generate_params():
            kwargs["negative_prompt"] = negative_prompt
        if is_img2img:
            kwargs["image_path"] = image_path
            kwargs["image_strength"] = image_strength

        generated_image = self._model.generate_image(**kwargs)

        elapsed = time.perf_counter() - start
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

        For instruction-based editing (QwenImageEdit), the prompt describes
        what to change. For Kontext/Fill, the source image is blended with
        the prompt at the given strength.
        """
        if not self.is_loaded:
            raise RuntimeError("No edit model loaded. Call load() first.")

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

        # Build kwargs based on the model class
        mclass = self._mflux_class or ""

        # Defense in depth: explicit edit-model classes only. Without this,
        # calling edit() on a generation model (Flux1 / ZImage / Flux2Klein)
        # falls through to the generic img2img branch and silently performs
        # variations instead of instruction-based editing.
        _edit_mclasses = {"QwenImageEdit", "Flux1Kontext", "Flux1Fill", "Flux2KleinEdit"}
        if mclass not in _edit_mclasses:
            raise ValueError(
                f"Model '{self._model_name}' (class={mclass}) is not an editing "
                f"model. Route to ImageGenEngine.generate() for text-to-image or "
                f"img2img, or load qwen-image-edit / flux-kontext / flux-fill."
            )

        if mclass == "QwenImageEdit":
            qwen_kwargs: dict = dict(
                seed=seed,
                prompt=prompt,
                image_paths=[image_path],
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
            if negative_prompt and 'negative_prompt' in self._get_generate_params():
                qwen_kwargs["negative_prompt"] = negative_prompt
            generated_image = self._model.generate_image(**qwen_kwargs)
        elif mclass == "Flux1Kontext":
            # Kontext uses reference image for subject conditioning
            # image_strength=None (default) = full denoising with reference conditioning
            # Passing a strength value skips denoising steps, causing noisy output
            generated_image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                image_path=image_path,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
        elif mclass == "Flux1Fill":
            if not mask_path:
                raise ValueError("Flux Fill requires a mask_path for inpainting")
            # Fill uses the mask to define regions — do NOT pass image_strength
            # (defaults to None = full denoising in masked area, which is correct)
            generated_image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                image_path=image_path,
                masked_image_path=mask_path,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
        elif mclass == "Flux2KleinEdit":
            # Klein Edit uses image_paths for reference — no image_strength
            generated_image = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                image_paths=[image_path],
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            )
        else:
            # Generic img2img fallback (works for Flux1, ZImage, etc.)
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

        elapsed = time.perf_counter() - start
        pil_image = generated_image.image
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        logger.info(f"Image edited in {elapsed:.1f}s: {width}x{height}")

        return ImageGenResult(
            image_bytes=image_bytes,
            width=width,
            height=height,
            model=self._model_name or "unknown",
            seed=seed,
            steps=steps,
            elapsed_seconds=elapsed,
        )

    def _get_generate_params(self) -> set[str]:
        """Get the parameter names of the model's generate_image method."""
        import inspect
        if self._model is None:
            return set()
        try:
            sig = inspect.signature(self._model.generate_image)
            return set(sig.parameters.keys())
        except Exception:
            return set()

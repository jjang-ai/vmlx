"""Per-request IMAGE preprocessing controls (the image counterpart of
``video_controls``).

Request fields (Chat, Responses, Anthropic, Ollama):

- ``image_max_pixels`` / ``image_min_pixels``: per-image pixel bounds, aspect
  preserved. The image is bounded BEFORE the processor sees it, in a
  request-local copy — the loaded processor is never mutated, so concurrent
  requests cannot see each other's controls.
- ``image_resized_height`` / ``image_resized_width``: an explicit size (both
  or neither); wins over the pixel bounds.
- ``image_token_budget``: Gemma-4 visual budget (its processor's
  ``max_soft_tokens``); on any other processor it is reported as unsupported
  through the response ``warnings`` — never silently ignored.

Precedence per image: explicit size > max/min pixels > the processor's own
defaults. Whatever this module does, the processor still applies its OWN
floor/ceiling afterwards (Qwen image processors: ``size.shortest_edge`` ≈
65,536 px minimum); when that changes the effective result the engine reports
the effective size and tokens in ``warnings`` (prefix ``image_controls:``).
With ``media_controls_strict=true`` such a deviation is a 4xx instead.

Every control is part of the media identity (pixel cache / prefix cache keys),
so two requests with the same bytes and different controls never share cached
tensors or tokens.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Mapping

IMAGE_CONTROL_FIELDS: tuple[str, ...] = (
    "image_max_pixels",
    "image_min_pixels",
    "image_resized_height",
    "image_resized_width",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_int(name: str, value: Any) -> int:
    if not _is_number(value) or not math.isfinite(float(value)) or float(value) < 1 or float(value) != int(value):
        raise ValueError(f"{name} must be a whole number >= 1")
    return int(value)


@dataclass(frozen=True)
class ImageControls:
    max_pixels: int | None = None
    min_pixels: int | None = None
    resized_height: int | None = None
    resized_width: int | None = None

    @property
    def is_unset(self) -> bool:
        return all(v is None for v in (self.max_pixels, self.min_pixels, self.resized_height, self.resized_width))

    @property
    def resize(self) -> tuple[int, int] | None:
        if self.resized_height is not None and self.resized_width is not None:
            return int(self.resized_height), int(self.resized_width)
        return None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None, *, validate: bool = True) -> "ImageControls":
        raw = {f: values.get(f) for f in IMAGE_CONTROL_FIELDS} if values else {}
        if validate:
            return validate_image_controls(raw)
        return cls(
            max_pixels=raw.get("image_max_pixels"),
            min_pixels=raw.get("image_min_pixels"),
            resized_height=raw.get("image_resized_height"),
            resized_width=raw.get("image_resized_width"),
        )

    @classmethod
    def from_request(cls, request: Any) -> "ImageControls":
        existing = getattr(request, "image_controls", None)
        if isinstance(existing, ImageControls):
            return existing
        return cls.from_mapping({f: getattr(request, f, None) for f in IMAGE_CONTROL_FIELDS}, validate=False)

    def fragment(self) -> str:
        """Cache-identity fragment (every control at its sent value)."""
        return (
            f"image_max_pixels={self.max_pixels if self.max_pixels is not None else '-'}:"
            f"image_min_pixels={self.min_pixels if self.min_pixels is not None else '-'}:"
            f"image_resized={self.resized_height if self.resized_height is not None else '-'}x"
            f"{self.resized_width if self.resized_width is not None else '-'}"
        )

    def as_request_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in IMAGE_CONTROL_FIELDS:
            v = getattr(self, f[len("image_"):])
            if v is not None:
                out[f] = v
        return out


def validate_image_controls(values: Mapping[str, Any] | None) -> ImageControls:
    """Validate request fields; raises ``ValueError`` naming the field."""
    values = values or {}
    out: dict[str, Any] = {}
    for f in ("image_max_pixels", "image_min_pixels", "image_resized_height", "image_resized_width"):
        v = values.get(f)
        out[f] = _positive_int(f, v) if v is not None else None
    if out["image_min_pixels"] is not None and out["image_max_pixels"] is not None and out["image_min_pixels"] > out["image_max_pixels"]:
        raise ValueError("image_min_pixels must be <= image_max_pixels")
    if (out["image_resized_height"] is None) != (out["image_resized_width"] is None):
        raise ValueError("image_resized_height and image_resized_width must be given together")
    if out["image_resized_height"] is not None and out["image_max_pixels"] is not None and out["image_resized_height"] * out["image_resized_width"] > out["image_max_pixels"]:
        raise ValueError("the explicit image size exceeds image_max_pixels")
    return ImageControls(
        max_pixels=out["image_max_pixels"],
        min_pixels=out["image_min_pixels"],
        resized_height=out["image_resized_height"],
        resized_width=out["image_resized_width"],
    )


def image_controls_from_kwargs(kwargs: Mapping[str, Any]) -> ImageControls | None:
    existing = kwargs.get("image_controls")
    if isinstance(existing, ImageControls):
        return existing
    controls = ImageControls.from_mapping(kwargs, validate=False)
    return None if controls.is_unset else controls


def pop_image_control_kwargs(kwargs: dict[str, Any]) -> ImageControls | None:
    controls = image_controls_from_kwargs(kwargs)
    for f in (*IMAGE_CONTROL_FIELDS, "image_controls"):
        kwargs.pop(f, None)
    return controls


def target_size(height: int, width: int, controls: ImageControls) -> tuple[int, int]:
    """(h, w) this module resizes to: explicit size wins; else scale DOWN to fit
    ``max_pixels`` and UP to reach ``min_pixels``, aspect preserved."""
    if controls.resize is not None:
        return max(1, controls.resize[0]), max(1, controls.resize[1])
    h, w = int(height), int(width)
    if controls.max_pixels and h * w > controls.max_pixels:
        s = math.sqrt(controls.max_pixels / float(h * w))
        h, w = max(1, int(math.floor(h * s))), max(1, int(math.floor(w * s)))
    if controls.min_pixels and h * w < controls.min_pixels:
        s = math.sqrt(controls.min_pixels / float(h * w))
        h, w = max(1, int(math.ceil(h * s))), max(1, int(math.ceil(w * s)))
    return h, w


def bound_image_file(path: str, controls: ImageControls, out_dir: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Write a bounded copy of the image for THIS request; returns
    (new_path_or_original, (h, w) before, (h, w) after). The original file and
    the processor are untouched."""
    from PIL import Image

    with Image.open(path) as im:
        w0, h0 = im.size
        h, w = target_size(h0, w0, controls)
        if (h, w) == (h0, w0):
            return path, (h0, w0), (h0, w0)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        new_path = os.path.join(out_dir, f"{base}-{h}x{w}.png")
        im.convert("RGB").resize((w, h), Image.BICUBIC if h * w > h0 * w0 else Image.LANCZOS).save(new_path, format="PNG")
    return new_path, (h0, w0), (h, w)


def image_processor_geometry(processor: Any) -> tuple[int, int | None, int | None]:
    """(pixels per image token, pixel floor, pixel ceiling) of the loaded image processor."""
    from .video_controls import image_pixel_floor, image_token_pixels

    ip = getattr(processor, "image_processor", None) or processor
    ceiling = None
    try:
        size = getattr(ip, "size", None)
        ceiling = getattr(ip, "max_pixels", None) or (size.get("longest_edge") if isinstance(size, dict) else None)
        ceiling = int(ceiling) if ceiling else None
    except Exception:
        ceiling = None
    return image_token_pixels(processor), image_pixel_floor(processor), ceiling


def image_controls_diagnostics(
    controls: ImageControls,
    *,
    before: tuple[int, int],
    after: tuple[int, int],
    token_pixels: int,
    pixel_floor: int | None,
    pixel_ceiling: int | None,
) -> tuple[list[str], list[str]]:
    """(reports, unmeetable): what the processor will still change after our
    bound, with effective values. ``unmeetable`` entries are the ones strict
    mode rejects."""
    from .video_controls import smart_resize_dims

    h, w = int(after[0]), int(after[1])
    factor = max(1, int(round(math.sqrt(max(1, int(token_pixels))))))
    eh, ew = smart_resize_dims(h, w, factor=factor, min_pixels=pixel_floor, max_pixels=pixel_ceiling)
    tokens = max(1, (eh * ew) // max(1, int(token_pixels)))
    reports: list[str] = []
    unmeetable: list[str] = []
    asked = f"{controls.resize[0]}x{controls.resize[1]}" if controls.resize else (f"image_max_pixels={controls.max_pixels}" if controls.max_pixels else f"image_min_pixels={controls.min_pixels}")
    if pixel_floor and h * w < int(pixel_floor):
        msg = (f"image_controls: {asked} bounded the image to {h}x{w}, below the image processor floor ({pixel_floor} px); "
               f"the processor upscales it: effective {eh}x{ew}, {tokens} tokens")
        reports.append(msg); unmeetable.append(msg)
    elif pixel_ceiling and h * w > int(pixel_ceiling):
        msg = (f"image_controls: {asked} leaves the image at {h}x{w}, above the image processor ceiling ({pixel_ceiling} px); "
               f"the processor downscales it: effective {eh}x{ew}, {tokens} tokens")
        reports.append(msg); unmeetable.append(msg)
    elif (eh, ew) != (h, w) and controls.resize is not None:
        reports.append(f"image_controls: explicit size {h}x{w} is rounded by the processor to its patch grid: effective {eh}x{ew}, {tokens} tokens")
    return reports, unmeetable


def image_token_budget_support(processor: Any) -> bool:
    """True when the loaded processor honours ``image_token_budget`` (Gemma 4)."""
    processor_type = type(processor)
    blob = " ".join(str(item) for item in (
        getattr(processor_type, "__module__", ""), getattr(processor_type, "__name__", ""),
        getattr(processor, "model_type", ""), getattr(getattr(processor, "config", None), "model_type", ""),
    )).lower()
    return "gemma4" in blob or "gemma_4" in blob

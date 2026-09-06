"""Per-request video preprocessing controls.

One normalized object carries every knob a caller may set for video input:

- ``fps`` / ``max_frames``: temporal sampling (how many frames, how often).
- ``max_pixels`` / ``min_pixels``: per-frame pixel budget after resizing,
  aspect preserved (the qwen-vl-utils contract mlx-vlm's ``fetch_video``
  implements: frames are resized with ``smart_resize`` to fit the budget,
  rounded to the processor's patch factor).
- ``total_pixels``: whole-clip budget; the loader derives a per-frame cap
  from it (``total_pixels / nframes * FRAME_FACTOR``).
- ``resized_height`` / ``resized_width``: explicit frame size (both or
  neither); wins over the pixel budgets.

The same object is validated once at the API edge (Chat, Responses,
Anthropic, Ollama), forwarded through the engine, applied to the native
loader and to the frame-fallback path, and folded into every cache key
(pixel cache, media digests, SimpleEngine source keys), so two requests
with the same bytes and different controls never share preprocessed
features. Unset controls mean "processor default"; the temporal defaults
are the engine's ``DEFAULT_FPS`` / ``MAX_FRAMES`` and are folded into the
key at their effective values, so leaving a control unset and spelling out
the default is the same key.

Known library limit (mlx-vlm 0.5 ``fetch_video``): a per-frame budget above
``MLX_VLM_VIDEO_MAX_PIXELS`` is clamped by the loader with a warning, and
the frame-fallback path has no such clamp. ``clamp_note`` reports the first
case so callers can say what actually applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

VIDEO_CONTROL_FIELDS: tuple[str, ...] = (
    "video_fps",
    "video_max_frames",
    "video_max_pixels",
    "video_min_pixels",
    "video_total_pixels",
    "video_resized_height",
    "video_resized_width",
    "video_token_budget",
)

# Qwen-style video geometry: one token per (image_factor x image_factor)
# pixel patch after merging, over pairs of frames (temporal patch 2). The
# loader's per-clip budget ``total_pixels`` therefore maps to tokens as
# tokens ~= total_pixels / image_factor**2 (see fetch_video: per-frame
# max_pixels = total_pixels / nframes * FRAME_FACTOR).
VIDEO_TOKEN_IMAGE_FACTOR = 28

# mlx-vlm 0.5 ``video_generate.VIDEO_MAX_PIXELS``: the loader clamps a larger
# per-frame budget to this value (768 * 28 * 28).
MLX_VLM_VIDEO_MAX_PIXELS = 602112


def _field_to_attr(field: str) -> str:
    return field[len("video_"):]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_int(name: str, value: Any) -> int:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite positive integer")
    if float(value) != int(value):
        raise ValueError(f"{name} must be a whole number")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


@dataclass(frozen=True)
class FallbackFrameBounds:
    """Resolved sizing for the sampled-frame (image) fallback path."""

    max_long_edge: int
    max_pixels: int | None
    resize: tuple[int, int] | None  # (height, width)


@dataclass(frozen=True)
class VideoControls:
    fps: float | None = None
    max_frames: int | None = None
    max_pixels: int | None = None
    min_pixels: int | None = None
    total_pixels: int | None = None
    resized_height: int | None = None
    resized_width: int | None = None
    # Whole-clip vision-token budget; derived into total_pixels for the loader
    # (approximate: smart_resize rounds to the patch factor and min_pixels
    # floors each frame).
    token_budget: int | None = None

    # ── construction ──────────────────────────────────────────────────
    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None, *, validate: bool = True) -> "VideoControls":
        """Build from request-field names (``video_fps`` ...). ``None`` values are unset."""
        raw = {f: values.get(f) for f in VIDEO_CONTROL_FIELDS} if values else {}
        return validate_video_controls(raw) if validate else cls(**{_field_to_attr(k): v for k, v in raw.items()})

    @classmethod
    def from_request(cls, request: Any, *, validate: bool = False) -> "VideoControls":
        """Build from an object carrying the request fields as attributes.

        A ``video_controls`` attribute, when present, is authoritative.
        """
        existing = getattr(request, "video_controls", None)
        if isinstance(existing, VideoControls):
            return existing
        raw = {f: getattr(request, f, None) for f in VIDEO_CONTROL_FIELDS}
        return validate_video_controls(raw) if validate else cls(**{_field_to_attr(k): v for k, v in raw.items()})

    # ── views ─────────────────────────────────────────────────────────
    @property
    def is_unset(self) -> bool:
        return all(getattr(self, _field_to_attr(f)) is None for f in VIDEO_CONTROL_FIELDS)

    @property
    def has_pixel_controls(self) -> bool:
        return any(
            v is not None
            for v in (self.max_pixels, self.min_pixels, self.total_pixels, self.resized_height, self.resized_width, self.token_budget)
        )

    def effective_total_pixels(self) -> int | None:
        """``total_pixels`` as sent, else the value derived from ``token_budget``."""
        if self.total_pixels is not None:
            return int(self.total_pixels)
        if self.token_budget is not None:
            return int(self.token_budget) * VIDEO_TOKEN_IMAGE_FACTOR * VIDEO_TOKEN_IMAGE_FACTOR
        return None

    def effective_fps(self) -> float:
        if self.fps is not None:
            return float(self.fps)
        from .models.mllm import DEFAULT_FPS

        return float(DEFAULT_FPS)

    def effective_max_frames(self) -> int:
        if self.max_frames is not None:
            return int(self.max_frames)
        from .models.mllm import MAX_FRAMES

        return int(MAX_FRAMES)

    def as_request_kwargs(self) -> dict[str, Any]:
        """Only the set fields, under their request names (for forwarding)."""
        out: dict[str, Any] = {}
        for f in VIDEO_CONTROL_FIELDS:
            v = getattr(self, _field_to_attr(f))
            if v is not None:
                out[f] = v
        return out

    def pixel_fragment(self) -> str:
        return (
            f"max_pixels={self.max_pixels if self.max_pixels is not None else '-'}"
            f":min_pixels={self.min_pixels if self.min_pixels is not None else '-'}"
            f":total_pixels={self.total_pixels if self.total_pixels is not None else '-'}"
            f":token_budget={self.token_budget if self.token_budget is not None else '-'}"
            f":resized={self.resized_height if self.resized_height is not None else '-'}x"
            f"{self.resized_width if self.resized_width is not None else '-'}"
        )

    def cache_key_fragment(self) -> str:
        """Every control at its EFFECTIVE value: unset temporal controls read
        as the engine defaults, so equal extraction means equal key."""
        return f"video_fps={self.effective_fps():g}:video_max_frames={self.effective_max_frames()}:" + self.pixel_fragment()

    # ── application ───────────────────────────────────────────────────
    def fetch_video_element(self, video_path: str) -> dict[str, Any]:
        """The ``ele`` dict for mlx-vlm ``video_generate.fetch_video``."""
        ele: dict[str, Any] = {
            "video": str(video_path),
            "fps": self.effective_fps(),
            "max_frames": self.effective_max_frames(),
        }
        if self.min_pixels is not None:
            ele["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            ele["max_pixels"] = int(self.max_pixels)
        total = self.effective_total_pixels()
        if total is not None:
            ele["total_pixels"] = total
        if self.resized_height is not None and self.resized_width is not None:
            ele["resized_height"] = int(self.resized_height)
            ele["resized_width"] = int(self.resized_width)
        return ele

    def fallback_bounds(self, *, default_long_edge: int) -> FallbackFrameBounds:
        """Sizing for the sampled-frame fallback: explicit size wins, else the
        per-frame pixel budget (aspect preserved), else the long-edge default."""
        resize = None
        if self.resized_height is not None and self.resized_width is not None:
            resize = (int(self.resized_height), int(self.resized_width))
        max_pixels = int(self.max_pixels) if self.max_pixels is not None else None
        total = self.effective_total_pixels()
        if max_pixels is None and total is not None:
            # Spread the clip budget over the frames the fallback will sample
            # (each sampled frame becomes one image, so no temporal pairing).
            max_pixels = max(1, int(total // max(1, self.effective_max_frames())))
        return FallbackFrameBounds(
            max_long_edge=int(default_long_edge),
            max_pixels=max_pixels,
            resize=resize,
        )

    def clamp_note(self) -> str | None:
        """What the native loader will NOT honour as sent, if anything."""
        if self.max_pixels is not None and int(self.max_pixels) > MLX_VLM_VIDEO_MAX_PIXELS:
            return (
                f"video_max_pixels={int(self.max_pixels)} exceeds the mlx-vlm per-frame limit "
                f"{MLX_VLM_VIDEO_MAX_PIXELS}; the native loader clamps to the limit"
            )
        return None


def validate_video_controls(values: Mapping[str, Any] | None) -> VideoControls:
    """Validate request fields and build the normalized object.

    Raises ``ValueError`` naming the offending field. Rules: fps finite and
    > 0; frame cap and pixel budgets whole numbers >= 1; ``min_pixels`` <=
    ``max_pixels`` when both set; ``resized_height``/``resized_width`` both or
    neither; an explicit size must fit an explicit ``max_pixels``.
    """
    values = values or {}
    fps = values.get("video_fps")
    if fps is not None:
        if not _is_number(fps) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("video_fps must be a finite number greater than 0")
        fps = float(fps)
    max_frames = values.get("video_max_frames")
    if max_frames is not None:
        max_frames = _positive_int("video_max_frames", max_frames)
    ints: dict[str, int | None] = {}
    for name in ("video_max_pixels", "video_min_pixels", "video_total_pixels", "video_resized_height", "video_resized_width", "video_token_budget"):
        v = values.get(name)
        ints[name] = _positive_int(name, v) if v is not None else None
    if (ints["video_resized_height"] is None) != (ints["video_resized_width"] is None):
        raise ValueError("video_resized_height and video_resized_width must be given together")
    if (
        ints["video_min_pixels"] is not None
        and ints["video_max_pixels"] is not None
        and ints["video_min_pixels"] > ints["video_max_pixels"]
    ):
        raise ValueError("video_min_pixels must not exceed video_max_pixels")
    if (
        ints["video_resized_height"] is not None
        and ints["video_max_pixels"] is not None
        and ints["video_resized_height"] * ints["video_resized_width"] > ints["video_max_pixels"]
    ):
        raise ValueError("video_resized_height x video_resized_width exceeds video_max_pixels")
    return VideoControls(
        fps=fps,
        max_frames=max_frames,
        max_pixels=ints["video_max_pixels"],
        min_pixels=ints["video_min_pixels"],
        total_pixels=ints["video_total_pixels"],
        resized_height=ints["video_resized_height"],
        resized_width=ints["video_resized_width"],
        token_budget=ints["video_token_budget"],
    )


def video_control_kwargs(source: Any) -> dict[str, Any]:
    """Forwardable kwargs (request names, set fields only) from a request
    model, a plain mapping, or a kwargs dict. Includes the normalized
    ``video_controls`` object so downstream hops need no re-parsing."""
    if isinstance(source, Mapping):
        controls = VideoControls.from_mapping(source, validate=False)
    else:
        controls = VideoControls.from_request(source)
    out = controls.as_request_kwargs()
    if out:
        out["video_controls"] = controls
    return out


def video_controls_from_kwargs(kwargs: Mapping[str, Any]) -> VideoControls | None:
    """The normalized object from engine kwargs (``video_controls`` if
    present, else rebuilt from the individual fields); None when nothing is set."""
    existing = kwargs.get("video_controls")
    if isinstance(existing, VideoControls):
        return existing
    controls = VideoControls.from_mapping(kwargs, validate=False)
    return None if controls.is_unset else controls


def pop_video_control_kwargs(kwargs: dict[str, Any]) -> VideoControls | None:
    """Remove every video control from a kwargs dict and return the object."""
    controls = video_controls_from_kwargs(kwargs)
    for f in (*VIDEO_CONTROL_FIELDS, "video_controls"):
        kwargs.pop(f, None)
    return controls


def bound_video_frames(
    frames: Iterable[Any],
    *,
    max_long_edge: int = 0,
    max_pixels: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[Any]:
    """Resize sampled frames (H x W x C arrays) for the image fallback path.

    An explicit ``resize`` (height, width) wins. Otherwise the frame is scaled
    down, aspect preserved, so it fits both ``max_pixels`` and
    ``max_long_edge`` (each ignored when unset/non-positive). Frames are never
    upscaled. Invalid/test sentinel frames are left untouched.
    """
    frames = list(frames)
    if resize is None and (max_long_edge is None or max_long_edge <= 0) and (max_pixels is None or max_pixels <= 0):
        return frames
    try:
        import cv2
    except Exception:
        return frames
    bounded: list[Any] = []
    for frame in frames:
        try:
            height, width = (int(frame.shape[0]), int(frame.shape[1]))
            if resize is not None:
                target_h, target_w = max(1, int(resize[0])), max(1, int(resize[1]))
                if (target_h, target_w) == (height, width):
                    bounded.append(frame)
                    continue
                interp = cv2.INTER_AREA if target_h * target_w < height * width else cv2.INTER_CUBIC
                bounded.append(cv2.resize(frame, (target_w, target_h), interpolation=interp))
                continue
            scale = 1.0
            if max_long_edge and max_long_edge > 0:
                scale = min(scale, max_long_edge / float(max(height, width)))
            if max_pixels and max_pixels > 0 and height * width > max_pixels:
                scale = min(scale, math.sqrt(max_pixels / float(height * width)))
            if scale >= 1.0:
                bounded.append(frame)
                continue
            # Round like the long-edge bound always did (pinned shapes), then
            # floor only if rounding overshoots an explicit pixel budget.
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            if max_pixels and max_pixels > 0 and resized_width * resized_height > max_pixels:
                resized_width = max(1, int(math.floor(width * scale)))
                resized_height = max(1, int(math.floor(height * scale)))
            bounded.append(cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA))
        except Exception:
            bounded.append(frame)
    return bounded

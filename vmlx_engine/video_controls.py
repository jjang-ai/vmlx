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

Scope of the token budget: it applies PER VIDEO ATTACHMENT (each video gets
its own clip budget), never to image attachments, never to output tokens or
context size, and it is an ESTIMATE, not a hard cap: frames round to the
patch factor, ``min_pixels`` floors each frame, and the transformers video
processor (mlx-vlm 0.5 delegates to Qwen3VLVideoProcessor) applies its own
whole-clip bounds (16,384 .. ~12.6 M pixels over all frames). The processed
grid the engine logs per request is the actual token count.

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

# Legacy (qwen-vl-utils) video geometry: one token per (image_factor x
# image_factor) pixel patch after merging. Only the fallback when the loaded
# processor does not expose its geometry: the real factor is read from the
# processor (``video_token_pixels``). Measured on Qwen3-VL (patch 16, merge 2,
# temporal patch 2): 16 frames of 352x640 -> grid 8x22x40 -> 1,760 tokens, i.e.
# 2,048 pixels per token, not 784 — a budget converted with 784 asked the
# processor for 2.6x too few pixels and could never be met.
VIDEO_TOKEN_IMAGE_FACTOR = 28
DEFAULT_VIDEO_TOKEN_PIXELS = VIDEO_TOKEN_IMAGE_FACTOR * VIDEO_TOKEN_IMAGE_FACTOR


def video_token_pixels(processor: Any = None) -> int:
    """Pixels of the sampled clip that one merged video token covers:
    ``temporal_patch_size * (patch_size * merge_size) ** 2`` from the loaded
    video processor, else the legacy 28x28."""
    vp = getattr(processor, "video_processor", None) if processor is not None else None
    if vp is None:
        vp = processor
    try:
        patch = int(getattr(vp, "patch_size", 0) or 0)
        merge = int(getattr(vp, "merge_size", 0) or 0)
        temporal = int(getattr(vp, "temporal_patch_size", 0) or 0)
    except (TypeError, ValueError):
        return DEFAULT_VIDEO_TOKEN_PIXELS
    if patch > 0 and merge > 0 and temporal > 0:
        return temporal * (patch * merge) ** 2
    return DEFAULT_VIDEO_TOKEN_PIXELS

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
class FallbackFramePlan:
    """How many sampled frames the image fallback keeps and how large each one
    is, so that a request's video-token budget is met on a path where every
    frame is a separate image with the image processor's OWN pixel floor.

    ``expected_tokens_per_frame`` is the image processor's smart-resize result
    for a frame bounded to ``per_frame_max_pixels`` (its pixel floor applies
    even when the budget asks for less), so ``expected_total`` is what the
    processor will actually emit for ``num_frames`` frames of this size.
    """

    num_frames: int
    per_frame_max_pixels: int | None
    expected_tokens_per_frame: int | None
    expected_total: int | None
    budget: int | None
    frame_cap: int
    reason: str

    @property
    def met(self) -> bool | None:
        if self.budget is None or self.expected_total is None:
            return None
        return self.expected_total <= self.budget


def image_token_pixels(processor: Any = None) -> int:
    """Pixels one merged IMAGE token covers: ``(patch_size * merge_size) ** 2``
    from the loaded image processor (no temporal pairing on the image path),
    else the legacy 28x28."""
    ip = getattr(processor, "image_processor", None) if processor is not None else None
    if ip is None:
        ip = processor
    try:
        patch = int(getattr(ip, "patch_size", 0) or 0)
        merge = int(getattr(ip, "merge_size", 0) or 0)
    except (TypeError, ValueError):
        return 28 * 28
    if patch > 0 and merge > 0:
        return (patch * merge) ** 2
    return 28 * 28


def image_pixel_floor(processor: Any = None) -> int | None:
    """The image processor's minimum pixels per image (``min_pixels`` or
    ``size.shortest_edge``), below which it UPSCALES; ``None`` when unknown."""
    ip = getattr(processor, "image_processor", None) if processor is not None else None
    if ip is None:
        ip = processor
    if ip is None:
        return None
    value = getattr(ip, "min_pixels", None)
    if value is None:
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            value = size.get("shortest_edge") or size.get("min_pixels")
    try:
        value = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return value if value and value > 0 else None


def smart_resize_dims(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    """The Qwen-VL image processor's ``smart_resize``: dims rounded to
    ``factor``, scaled DOWN to fit ``max_pixels`` or UP to reach ``min_pixels``."""
    factor = max(1, int(factor))
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)
    if max_pixels is not None and max_pixels > 0 and h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / float(max_pixels))
        h_bar = max(factor, int(math.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(math.floor(width / beta / factor)) * factor)
    elif min_pixels is not None and min_pixels > 0 and h_bar * w_bar < min_pixels:
        beta = math.sqrt(float(min_pixels) / (height * width))
        h_bar = int(math.ceil(height * beta / factor)) * factor
        w_bar = int(math.ceil(width * beta / factor)) * factor
    return h_bar, w_bar


def _bounded_frame_dims(height: int, width: int, max_pixels: int | None) -> tuple[int, int]:
    """Dims ``bound_video_frames`` produces for a per-frame pixel bound (never upscales)."""
    if not max_pixels or max_pixels <= 0 or height * width <= max_pixels:
        return height, width
    scale = math.sqrt(max_pixels / float(height * width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if resized_width * resized_height > max_pixels:
        resized_width = max(1, int(math.floor(width * scale)))
        resized_height = max(1, int(math.floor(height * scale)))
    return resized_height, resized_width


def fallback_frame_tokens(
    height: int,
    width: int,
    *,
    per_frame_max_pixels: int | None,
    token_pixels: int,
    pixel_floor: int | None,
    pixel_ceiling: int | None = None,
) -> int:
    """Image tokens the processor emits for one sampled frame bounded to
    ``per_frame_max_pixels`` (its floor/ceiling applied after our bound)."""
    factor = max(1, int(round(math.sqrt(max(1, int(token_pixels))))))
    h, w = _bounded_frame_dims(int(height), int(width), per_frame_max_pixels)
    h_bar, w_bar = smart_resize_dims(h, w, factor=factor, min_pixels=pixel_floor, max_pixels=pixel_ceiling)
    return max(1, (h_bar * w_bar) // max(1, int(token_pixels)))


def plan_fallback_frames(
    controls: "VideoControls",
    *,
    frames_available: int,
    frame_cap: int,
    frame_height: int | None,
    frame_width: int | None,
    token_pixels: int,
    pixel_floor: int | None,
    pixel_ceiling: int | None = None,
) -> FallbackFramePlan:
    """Choose the frame count and per-frame pixel bound for the image fallback.

    ``frame_cap`` is the hard per-video ceiling (the request's image limit
    split across its videos, and the request's ``video_max_frames``). With a
    token budget (or ``total_pixels``) the plan keeps the LARGEST frame count
    ``n <= cap`` whose ``n x tokens(total / n)`` fits, where ``tokens`` is what
    the image processor emits after its own pixel floor. When even one frame
    at the floor exceeds the budget, one frame is kept and ``met`` is False
    (the floor is the processor's, reported honestly, never silently skipped).
    Without a budget the controls' explicit ``max_pixels`` (if any) applies.
    """
    cap = max(1, min(int(frames_available), int(frame_cap))) if frames_available > 0 else 0
    explicit = int(controls.max_pixels) if controls.max_pixels is not None else None
    total = controls.effective_total_pixels()
    budget = int(controls.token_budget) if controls.token_budget is not None else None
    if cap == 0:
        return FallbackFramePlan(0, explicit, None, None, budget, int(frame_cap), "no frames")
    if total is None:
        return FallbackFramePlan(cap, explicit, None, None, budget, int(frame_cap), "no budget")
    if explicit is not None and budget is None:
        return FallbackFramePlan(cap, explicit, None, None, budget, int(frame_cap), "explicit max_pixels")
    if explicit is not None:
        # BOTH an explicit per-frame pixel bound and a token budget: the bound
        # fixes each frame's size, the budget decides how many frames fit.
        # Live 2026-09-07 (27B fallback, strict): this branch used to return
        # "explicit max_pixels" with no token estimate, so a budget of 2 was
        # accepted with 8 full frames and strict mode had nothing to reject.
        if frame_height is None or frame_width is None or frame_height <= 0 or frame_width <= 0:
            return FallbackFramePlan(cap, explicit, None, None, budget, int(frame_cap), "explicit max_pixels; frame size unknown")
        tokens = fallback_frame_tokens(
            frame_height, frame_width, per_frame_max_pixels=explicit, token_pixels=token_pixels,
            pixel_floor=pixel_floor, pixel_ceiling=pixel_ceiling,
        )
        n = min(cap, max(0, int(budget) // max(1, tokens)))
        if n < 1:
            return FallbackFramePlan(
                1, explicit, tokens, tokens, budget, int(frame_cap),
                f"budget below one frame at the explicit max_pixels ({explicit} px -> {tokens} tokens/frame); one frame kept",
            )
        reason = "budget met at the explicit max_pixels" if n == cap else f"frames reduced {cap}->{n} to fit the budget at the explicit max_pixels"
        return FallbackFramePlan(n, explicit, tokens, n * tokens, budget, int(frame_cap), reason)
    # a video budget was spent in units of the VIDEO processor's pixels per
    # token (temporal pairs); each fallback frame is a single image
    video_token_px = int(controls.token_pixels or DEFAULT_VIDEO_TOKEN_PIXELS)
    budget_tokens = budget if budget is not None else max(1, int(total // max(1, video_token_px)))
    image_total = budget_tokens * max(1, int(token_pixels))
    if frame_height is None or frame_width is None or frame_height <= 0 or frame_width <= 0:
        per = max(1, int(image_total // cap))
        return FallbackFramePlan(cap, per, None, None, budget, int(frame_cap), "frame size unknown; even split")
    best: tuple[int, int, int] | None = None
    for n in range(cap, 0, -1):
        per = max(1, int(image_total // n))
        tokens = fallback_frame_tokens(
            frame_height, frame_width, per_frame_max_pixels=per, token_pixels=token_pixels,
            pixel_floor=pixel_floor, pixel_ceiling=pixel_ceiling,
        )
        if n * tokens <= budget_tokens:
            best = (n, per, tokens)
            break
    if best is None:
        per = max(1, int(image_total))
        tokens = fallback_frame_tokens(
            frame_height, frame_width, per_frame_max_pixels=per, token_pixels=token_pixels,
            pixel_floor=pixel_floor, pixel_ceiling=pixel_ceiling,
        )
        return FallbackFramePlan(
            1, per, tokens, tokens, budget, int(frame_cap),
            f"budget below the image processor floor ({pixel_floor} px -> {tokens} tokens/frame); one frame kept",
        )
    n, per, tokens = best
    reason = "budget met" if n == cap else f"frames reduced {cap}->{n} to fit the budget at the processor floor"
    return FallbackFramePlan(n, per, tokens, n * tokens, budget, int(frame_cap), reason)


def subsample_frames_evenly(frames: list[Any], keep: int) -> list[Any]:
    """Keep ``keep`` frames spread evenly over the sampled clip (first and last kept)."""
    n = len(frames)
    keep = max(0, min(int(keep), n))
    if keep == n or keep == 0:
        return list(frames[:keep])
    if keep == 1:
        return [frames[0]]
    idx = sorted({int(round(i * (n - 1) / (keep - 1))) for i in range(keep)})
    return [frames[i] for i in idx]


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
    # AND the processor (approximate: smart_resize rounds to the patch factor and min_pixels
    # floors each frame).
    token_budget: int | None = None
    # pixels per video token of the processor that will consume the clip;
    # set by the generator (``video_token_pixels``), legacy 784 otherwise
    token_pixels: int | None = None

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
        """``total_pixels`` as sent, else ``token_budget`` x the processor's
        pixels per token (``token_pixels``; legacy 28x28 when unknown)."""
        if self.total_pixels is not None:
            return int(self.total_pixels)
        if self.token_budget is not None:
            per_token = int(self.token_pixels or DEFAULT_VIDEO_TOKEN_PIXELS)
            return int(self.token_budget) * per_token
        return None

    def with_processor(self, processor: Any) -> "VideoControls":
        """A copy carrying the processor's pixels-per-token factor."""
        from dataclasses import replace

        return replace(self, token_pixels=video_token_pixels(processor))

    def processor_video_kwargs(self, processor: Any) -> dict[str, Any] | None:
        """Request-local ``videos_kwargs`` for the transformers video processor.

        The Qwen3-VL video processor sizes the WHOLE sampled clip from
        ``size.longest_edge`` (max total pixels) and ``size.shortest_edge`` (min
        total pixels) — exactly the per-video budget contract — so a
        ``total_pixels`` / ``token_budget`` request is enforced there instead of
        being approximated by the loader's per-frame cap (which floors at the
        loader's own minimum: budgets 512/256/128 all came out as 728 tokens
        before this). Returns None when the request carries no clip budget.
        Never mutates the processor."""
        total = self.effective_total_pixels()
        if total is None or total <= 0:
            return None
        vp = getattr(processor, "video_processor", None) or processor
        size = getattr(vp, "size", None) or {}
        try:
            default_short = int(size.get("shortest_edge") or 0) if isinstance(size, dict) else int(getattr(size, "shortest_edge", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            default_short = 0
        shortest = default_short if default_short > 0 else 1
        if shortest > total:
            shortest = int(total)
        return {"size": {"shortest_edge": int(shortest), "longest_edge": int(total)}}

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

    def fallback_bounds(
        self, *, default_long_edge: int, num_frames: int | None = None, per_frame_max_pixels: int | None = None
    ) -> FallbackFrameBounds:
        """Sizing for the sampled-frame fallback: explicit size wins, else the
        per-frame pixel budget (aspect preserved), else the long-edge default.
        ``per_frame_max_pixels`` (from ``plan_fallback_frames``) wins over the
        even split; ``num_frames`` is the count actually sampled (the split
        used the frame CAP before, so a 16-frame clip under a 128-frame cap got
        an eighth of its budget)."""
        resize = None
        if self.resized_height is not None and self.resized_width is not None:
            resize = (int(self.resized_height), int(self.resized_width))
        max_pixels = int(self.max_pixels) if self.max_pixels is not None else None
        total = self.effective_total_pixels()
        if max_pixels is None and per_frame_max_pixels is not None:
            max_pixels = max(1, int(per_frame_max_pixels))
        elif max_pixels is None and total is not None:
            # Spread the clip budget over the frames the fallback sampled
            # (each sampled frame becomes one image, so no temporal pairing).
            frames = int(num_frames) if num_frames else self.effective_max_frames()
            max_pixels = max(1, int(total // max(1, frames)))
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
    # image controls and the strict flag ride the same hop
    from .image_controls import ImageControls

    image = ImageControls.from_mapping(source, validate=False) if isinstance(source, Mapping) else ImageControls.from_request(source)
    if not image.is_unset:
        out.update(image.as_request_kwargs())
        out["image_controls"] = image
    strict = source.get("media_controls_strict") if isinstance(source, Mapping) else getattr(source, "media_controls_strict", None)
    if strict:
        out["media_controls_strict"] = True
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


def clip_budget_dims(
    num_frames: int,
    height: int,
    width: int,
    *,
    total_pixels: int,
    min_total_pixels: int = 0,
    factor: int = 32,
    temporal: int = 2,
) -> tuple[int, int]:
    """Frame size (h, w) that keeps the whole sampled clip within
    ``total_pixels`` under the Qwen3-VL processor's own rounding (frame edges
    to multiples of ``factor`` = patch x merge, frames to multiples of
    ``temporal``), i.e. transformers' video ``smart_resize``. Returns the
    input size unchanged when it already fits.

    Why here and not in the processor: the mlx-vlm Qwen3-VL processor this
    engine runs ignores per-call ``videos_kwargs`` (measured: size and
    do_sample_frames had no effect), so the budget is applied to the sampled
    frames before they reach it. Measured: 16 frames of 364x644 under a
    1,048,576-pixel budget -> 192x320 -> grid 8x12x20 -> 480 tokens, exactly
    what the transformers processor produced from the same size request."""
    import math

    num_frames = max(1, int(num_frames)); height = max(1, int(height)); width = max(1, int(width))
    t_bar = max(temporal, math.ceil(num_frames / temporal) * temporal)
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if total_pixels and t_bar * h_bar * w_bar > total_pixels:
        beta = math.sqrt((num_frames * height * width) / float(total_pixels))
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif min_total_pixels and t_bar * h_bar * w_bar < min_total_pixels:
        beta = math.sqrt(min_total_pixels / float(num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def clip_budget_factor(processor: Any = None) -> tuple[int, int]:
    """(spatial factor = patch x merge, temporal patch) of the loaded video
    processor; (32, 2) for Qwen3-VL, (28, 2) legacy."""
    vp = getattr(processor, "video_processor", None) if processor is not None else None
    if vp is None:
        vp = processor
    try:
        patch = int(getattr(vp, "patch_size", 0) or 0); merge = int(getattr(vp, "merge_size", 0) or 0); temporal = int(getattr(vp, "temporal_patch_size", 0) or 0)
    except (TypeError, ValueError):
        return VIDEO_TOKEN_IMAGE_FACTOR, 2
    if patch > 0 and merge > 0 and temporal > 0:
        return patch * merge, temporal
    return VIDEO_TOKEN_IMAGE_FACTOR, 2


# ── effective-settings reporting (best-effort contract) ─────────────────────
# The engine never refuses a video because its controls cannot be met exactly;
# it does the closest thing the processor allows and REPORTS the effective
# settings in the response ``warnings`` (prefix ``video_controls:``). These
# builders produce those messages; an empty return means nothing to report
# (the request was honoured as sent).

def fallback_plan_diagnostics(
    plan: "FallbackFramePlan",
    *,
    sampled: int,
    pixel_floor: int | None,
    resize: tuple[int, int] | None = None,
    token_pixels: int | None = None,
) -> list[str]:
    """Messages for the sampled-frame (image) fallback path."""
    out: list[str] = []
    floor_txt = f"{pixel_floor} px" if pixel_floor else "unknown"
    if plan.budget is not None and plan.met is False:
        out.append(
            f"video_controls: video_token_budget={plan.budget} cannot be met on the frame fallback: one frame at the "
            f"image processor floor ({floor_txt}) is {plan.expected_tokens_per_frame} tokens; effective {plan.num_frames} "
            f"frame(s), {plan.expected_total} media tokens (best effort, request not rejected)"
        )
    elif plan.budget is not None and plan.num_frames < min(int(sampled), int(plan.frame_cap)):
        out.append(
            f"video_controls: frame fallback kept {plan.num_frames} of {sampled} sampled frames to meet "
            f"video_token_budget={plan.budget} at the image processor floor ({floor_txt}, {plan.expected_tokens_per_frame} "
            f"tokens/frame): effective {plan.num_frames} frames x {plan.per_frame_max_pixels} px, {plan.expected_total} media tokens"
        )
    if int(sampled) > int(plan.frame_cap):
        out.append(
            f"video_controls: {sampled} sampled frames exceed the per-video frame cap {plan.frame_cap} (the request's "
            f"image limit shared by its videos); effective {plan.num_frames} frames, spread evenly"
        )
    if resize is not None and pixel_floor and token_pixels:
        h, w = int(resize[0]), int(resize[1])
        if h * w < int(pixel_floor):
            factor = max(1, int(round(math.sqrt(max(1, int(token_pixels))))))
            eh, ew = smart_resize_dims(h, w, factor=factor, min_pixels=int(pixel_floor), max_pixels=None)
            out.append(
                f"video_controls: explicit size {h}x{w} is below the image processor floor ({pixel_floor} px); "
                f"each frame is upscaled by the processor: effective {eh}x{ew}, {(eh * ew) // int(token_pixels)} tokens/frame"
            )
    return out


def clip_budget_diagnostics(
    controls: "VideoControls",
    *,
    num_frames: int,
    height: int,
    width: int,
    resized: tuple[int, int],
    factor: int,
    temporal: int,
) -> list[str]:
    """Messages for the native clip path (whole sampled clip under one budget)."""
    out: list[str] = []
    total = controls.effective_total_pixels()
    if not total:
        return out
    frames = max(1, int(num_frames))
    t_bar = max(temporal, math.ceil(frames / temporal) * temporal)
    h_bar, w_bar = int(resized[0]), int(resized[1])
    tokens = (h_bar // factor) * (w_bar // factor) * (t_bar // temporal)
    budget = int(controls.token_budget) if controls.token_budget is not None else None
    min_pixels = int(controls.min_pixels) if controls.min_pixels is not None else None
    if min_pixels and min_pixels * frames > int(total):
        out.append(
            f"video_controls: video_min_pixels={min_pixels} x {frames} frames ({min_pixels * frames} px) exceeds the clip "
            f"budget ({int(total)} px{f', video_token_budget={budget}' if budget is not None else ''}); the budget wins: "
            f"clip resized {height}x{width} -> {h_bar}x{w_bar}, {tokens} media tokens"
        )
    if budget is not None and tokens > budget:
        out.append(
            f"video_controls: video_token_budget={budget} cannot be met: the smallest clip grid is {factor}x{factor} per "
            f"frame pair ({t_bar // temporal} pairs for {frames} frames = {tokens} tokens); effective {h_bar}x{w_bar} frames, "
            f"{tokens} media tokens (best effort, request not rejected)"
        )
    return out

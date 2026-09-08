# Per-request media controls (video and image)

vMLX accepts per-request preprocessing controls for video and image inputs on
every dialect it serves: OpenAI Chat Completions and Responses, the Anthropic
Messages route, and the Ollama chat route (top level or inside `options`).
Controls are request-local: the loaded processor is never mutated, so
concurrent requests cannot influence each other, and every control is part of
the media identity used by the pixel cache and the prefix cache, so the same
bytes with different controls never share cached tensors or tokens.

## Video

| field | meaning |
|---|---|
| `video_fps` | sampling rate of the clip (frames per second) |
| `video_max_frames` | cap on sampled frames |
| `video_max_pixels` / `video_min_pixels` | per-frame pixel bounds |
| `video_total_pixels` | clip-wide pixel budget |
| `video_token_budget` | clip-wide token budget (converted with the processor's pixels-per-token) |
| `video_resized_height` / `video_resized_width` | explicit frame size (both or neither) |

Precedence per frame: explicit size > per-frame `max_pixels` > the clip
When an explicit per-frame `max_pixels` and a `video_token_budget` are both set on the frame-fallback path, the explicit size is kept per frame and the budget bounds the number of frames kept; a budget below one frame at that size is unmeetable (strict mode rejects it; best effort keeps one frame and reports it).
budget (`video_token_budget` or `video_total_pixels`, which wins over
`video_min_pixels` when they conflict) > the engine defaults.

Two runtime paths exist. Native video processors (Qwen3-VL style) receive the
sampled clip; the clip is resized to the budget with the processor's own
rounding before it is encoded. Frame-fallback families (qwen3_5 27B, step3p7,
gemma4) present each sampled frame as one image: the request's image limit
is split across its videos, and a token budget is met by keeping fewer,
floor-sized frames (subsampled evenly, first and last kept).

## Image

| field | meaning |
|---|---|
| `image_max_pixels` / `image_min_pixels` | per-image pixel bounds, aspect preserved |
| `image_resized_height` / `image_resized_width` | explicit size (both or neither) |
| `image_token_budget` | Gemma 4 visual budget (70, 140, 280, 560, 1120); reported as unsupported on any other processor |

Precedence per image: explicit size > `max_pixels` / `min_pixels` > the
processor's defaults. The bound is applied to a request-local copy of the
image before the processor sees it.

## What the processor still does afterwards

Every processor applies its own floor, ceiling and patch-grid rounding after
the request's bound (for example the Qwen image processor's minimum of
65,536 pixels, or the video processor's 32-pixel grid and two-frame temporal
pairing). The engine therefore reports what actually happened.

## Best-effort by default, strict on request

Image bounds are pre-aligned to the processor's patch grid (32 px on the
Qwen image processors), so an honoured `image_max_pixels` survives the
processor's own rounding. When the sent interval admits no grid-aligned size
at the image's aspect ratio (for example `image_min_pixels` equal to
`image_max_pixels`), the engine bounds to the largest size under the maximum
and reports the miss; strict mode rejects it.

Derived image copies are request-owned temporary files, removed as soon as
preprocessing returns, is rejected or raises.

By default a control that cannot be honoured as sent is never refused. The
engine does the closest thing the processor allows and reports the effective
settings in the response `warnings` array (Chat, Responses, Anthropic, Ollama)
and in the desktop app's chat bubble as a warning box. Messages are prefixed
`video_controls:` or `image_controls:` and name the effective frame count,
resolution and media-token usage, and any frames dropped, for example:

```
video_controls: frame fallback kept 7 of 16 sampled frames to meet
video_token_budget=512 at the image processor floor (65536 px, 66
tokens/frame): effective 7 frames x 74898 px, 462 media tokens
```

Nothing is reported when the request was honoured exactly.

With `media_controls_strict: true` the same situations are rejected instead,
with HTTP 400 and error code `media_controls_unmeetable` naming the floor or
the unsupported control. Streaming lanes emit the error as the stream's error
event.

Reduced sampling can legitimately miss content: a budget that keeps three
frames of an eight-screen clip may lose a screen. Transport success is not
unchanged understanding; the warnings exist so the caller can tell.

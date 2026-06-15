"""MiniMax-M3 vision preprocessing for the text-routed (SingleBatchGenerator) path.

ADDITIVE + GATED. Every entry point here is reached only when the env flag
``VMLX_M3_VL`` is set (see :func:`m3_vl_enabled`). When the flag is unset the
engine never imports/uses these helpers, so text-only M3 behavior is byte-for-byte
unchanged.

The standalone diagnostics (``diag_m3_vl_integrated.py``) proved that

    model(input_ids, cache=cache, pixel_values=pv, image_grid_thw=grid)

produces a coherent image description. The job of this module is to reproduce the
*exact* preprocessing those diagnostics used (MiniMax ``AutoProcessor``,
trust_remote_code), so the engine feeds the model identical
``input_ids`` / ``pixel_values`` / ``image_grid_thw`` tensors.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "on", "yes"}

# Process-wide processor cache, keyed by model path. AutoProcessor.from_pretrained
# is expensive; the processor is stateless across requests so caching is safe.
_PROCESSOR_CACHE: dict[str, Any] = {}


def m3_vl_enabled() -> bool:
    """True iff the VMLX_M3_VL gate is set. The ONLY switch for this whole path."""
    return os.environ.get("VMLX_M3_VL", "").strip().lower() in _TRUE


def is_m3_vl_model(model: Any) -> bool:
    """True iff `model` is a MiniMax-M3 build that carries a vision stack.

    Detection is structural (model_type contains minimax_m3 AND a `.vision`
    submodule exists), robust to loader wrappers.
    """
    try:
        mt = str(getattr(model, "model_type", "")).lower()
        if "minimax_m3" not in mt:
            # Some wrappers expose the inner model
            inner = getattr(model, "model", None)
            mt = str(getattr(inner, "model_type", "")).lower()
        return "minimax_m3" in mt and hasattr(model, "vision")
    except Exception:
        return False


def _get_processor(model_path: str):
    proc = _PROCESSOR_CACHE.get(model_path)
    if proc is not None:
        return proc
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _PROCESSOR_CACHE[model_path] = proc
    return proc


def _load_pil_images(images: List[Any]):
    """Resolve image inputs (paths/URLs/base64/OpenAI dicts) to PIL.Image (RGB)."""
    from PIL import Image

    from ..mllm import process_image_input

    out = []
    for img in images:
        path = process_image_input(img)
        out.append(Image.open(path).convert("RGB"))
    return out


def _normalize_messages_for_template(messages: List[dict]) -> Tuple[List[dict], List[Any]]:
    """Return (templated-ready messages, ordered raw image inputs).

    The MiniMax chat template emits an image placeholder for content items of
    the form ``{"type": "image"}``. OpenAI-format requests carry images as
    ``{"type": "image_url", "image_url": {"url": ...}}`` (or a bare string).
    Rewrite those to ``{"type": "image"}`` so the template renders the
    placeholder, and collect the raw image inputs in document order so they line
    up 1:1 with the placeholders.
    """
    # Keep ONLY the images from the most-recent image-bearing turn. A chat thread
    # re-sends prior-turn images in the message history every turn; M3's single-image
    # vision stack blends multiple concatenated images (e.g. a red then a blue image
    # -> "purple"), so collecting the whole history corrupts the answer. Using just the
    # latest image-bearing turn preserves "upload an image, then ask a follow-up" (that
    # turn stays the latest image-bearing one) while a NEW image replaces the old one.
    def _content_has_image(content) -> bool:
        return isinstance(content, list) and any(
            isinstance(it, dict) and it.get("type") in ("image_url", "image", "input_image")
            for it in content
        )

    last_img_idx = -1
    for i, msg in enumerate(messages):
        if _content_has_image(msg.get("content")):
            last_img_idx = i

    out_msgs: List[dict] = []
    raw_images: List[Any] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            out_msgs.append(msg)
            continue
        new_content = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            itype = item.get("type")
            if itype in ("image_url", "image", "input_image"):
                if i != last_img_idx:
                    # Drop stale prior-turn images entirely (no placeholder, no raw).
                    continue
                src = item.get("image_url", item.get("image", item.get("url")))
                if isinstance(src, dict):
                    src = src.get("url", src)
                if src is not None:
                    raw_images.append(src)
                new_content.append({"type": "image"})
            else:
                new_content.append(item)
        out_msgs.append({**msg, "content": new_content})
    return out_msgs, raw_images


def preprocess_m3_vl_messages(
    model_path: str,
    messages: List[dict],
    *,
    extra_images: Optional[List[Any]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> Optional[Tuple[List[int], Any, Any]]:
    """Template `messages` + run the MiniMax processor -> (input_ids, pv, grid).

    Mirrors the proven diag preprocessing exactly: normalize image items to
    ``{"type": "image"}``, ``apply_chat_template`` via the processor's tokenizer,
    then call the processor with the raw PIL images. Returns ``None`` when there
    are no images.
    """
    import mlx.core as mx
    import numpy as np

    proc = _get_processor(model_path)
    tok = getattr(proc, "tokenizer", proc)

    norm_msgs, raw_images = _normalize_messages_for_template(messages)
    if not raw_images and extra_images:
        # The server's extract_multimodal_content() flattens message content to
        # plain text and hands the images out-of-band (engine.chat images=...).
        # In that case the templated messages carry no image items, so inject
        # one {"type":"image"} placeholder per extra image into the LAST user
        # turn (matching the diag layout: image(s) precede the text). This is the
        # path exercised by the real /v1/chat/completions server flow.
        #
        # MiniMax-M3's vision stack handles ONE image: a multi-turn chat thread
        # re-sends every prior-turn image here, and concatenating them makes M3
        # blend (red + blue -> "purple"). extract_multimodal_content yields images
        # in document order, so the LAST entry is the most-recent turn's image —
        # keep only that one so a new image replaces the old instead of blending.
        raw_images = list(extra_images)[-1:]
        placeholders = [{"type": "image"} for _ in raw_images]
        injected = False
        for i in range(len(norm_msgs) - 1, -1, -1):
            if norm_msgs[i].get("role") == "user":
                m = norm_msgs[i]
                c = m.get("content")
                if isinstance(c, str):
                    new_c = placeholders + [{"type": "text", "text": c}]
                elif isinstance(c, list):
                    new_c = placeholders + list(c)
                else:
                    new_c = placeholders
                norm_msgs[i] = {**m, "content": new_c}
                injected = True
                break
        if not injected:
            norm_msgs = norm_msgs + [{"role": "user", "content": placeholders}]
    if not raw_images:
        return None

    tmpl_kwargs = {}
    if enable_thinking is not None:
        tmpl_kwargs["enable_thinking"] = enable_thinking
    try:
        txt = tok.apply_chat_template(
            norm_msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **tmpl_kwargs,
        )
    except TypeError:
        # Template doesn't accept enable_thinking — retry without it.
        txt = tok.apply_chat_template(
            norm_msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    pil_images = _load_pil_images(raw_images)
    if not pil_images:
        raise ValueError("M3 VL: all image inputs failed to load")

    out = proc(text=[txt], images=pil_images, return_tensors="np")
    ids = np.asarray(out["input_ids"][0]).astype(np.int64)
    input_ids = [int(x) for x in ids.tolist()]
    pixel_values = mx.array(out["pixel_values"]).astype(mx.bfloat16)
    grid = mx.array(np.asarray(out["image_grid_thw"]).astype(np.int32))
    # Materialize NOW so no lazy graph crosses threads. Preprocessing runs on the
    # server event loop thread; the forward runs on the scheduler worker thread.
    # A lazy astype bound to this thread's default stream would otherwise fail to
    # resolve there (P0 VL stream bug: "no Stream(gpu,0) in current thread").
    mx.eval(pixel_values, grid)

    n_img = int((ids == 200025).sum())
    logger.info(
        "M3 VL preprocess: %d tokens, %d image tokens, pixel_values=%s grid=%s",
        len(input_ids),
        n_img,
        tuple(pixel_values.shape),
        tuple(grid.shape),
    )
    if n_img == 0:
        raise ValueError(
            "M3 VL: chat template produced no image tokens (placeholder not "
            "rendered). Refusing to silently drop the image."
        )
    return input_ids, pixel_values, grid


def preprocess_m3_vl(

    model_path: str,
    prompt: str,
    images: List[Any],
) -> Optional[Tuple[List[int], Any, Any]]:
    """Run the MiniMax processor on a *templated* prompt + images.

    Returns ``(input_ids, pixel_values, image_grid_thw)`` where:
      - ``input_ids`` is a Python list[int] including the expanded image tokens
        (id 200025) at the image placeholder positions,
      - ``pixel_values`` is an ``mx.array`` (bfloat16) ready for the vision stack,
      - ``image_grid_thw`` is an ``mx.array`` (int32).

    Returns ``None`` when there are no images (caller falls back to text path).
    Raises on genuine processing failure (no silent papering-over).

    NOTE: ``prompt`` MUST already be the chat-templated string. The MiniMax
    template renders an ``<image>`` placeholder per image item; the processor
    expands each into the configured number of image tokens.
    """
    import mlx.core as mx
    import numpy as np

    if not images:
        return None

    proc = _get_processor(model_path)
    pil_images = _load_pil_images(images)
    if not pil_images:
        raise ValueError("M3 VL: all image inputs failed to load")

    out = proc(text=[prompt], images=pil_images, return_tensors="np")
    ids = np.asarray(out["input_ids"][0]).astype(np.int64)
    input_ids = [int(x) for x in ids.tolist()]
    pixel_values = mx.array(out["pixel_values"]).astype(mx.bfloat16)
    grid = mx.array(np.asarray(out["image_grid_thw"]).astype(np.int32))
    # Materialize NOW so no lazy graph crosses threads. Preprocessing runs on the
    # server event loop thread; the forward runs on the scheduler worker thread.
    # A lazy astype bound to this thread's default stream would otherwise fail to
    # resolve there (P0 VL stream bug: "no Stream(gpu,0) in current thread").
    mx.eval(pixel_values, grid)

    n_img = int((ids == 200025).sum())
    logger.info(
        "M3 VL preprocess: %d tokens, %d image tokens, pixel_values=%s grid=%s",
        len(input_ids),
        n_img,
        tuple(pixel_values.shape),
        tuple(grid.shape),
    )
    return input_ids, pixel_values, grid

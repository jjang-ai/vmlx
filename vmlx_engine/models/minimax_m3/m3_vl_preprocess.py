"""MiniMax-M3 vision preprocessing for the text-routed (SingleBatchGenerator) path.

ADDITIVE + GATED. Every entry point here is reached only when the env flag
``VMLX_M3_VL`` is set (see :func:`m3_vl_enabled`). When the flag is unset the
engine never imports/uses these helpers, so text-only M3 behavior is byte-for-byte
unchanged.

The standalone diagnostics (``diag_m3_vl_integrated.py``) proved that

    model(input_ids, cache=cache, pixel_values=pv, image_grid_thw=grid)

produces a coherent image description. This module keeps the same token contract
without importing MiniMax's HuggingFace ``AutoProcessor`` at request time. That
processor depends on torch/torchvision/video validation, which is unnecessary for
the text-routed vMLX image path and can block server requests on Apple Silicon.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "on", "yes"}

# Process-wide tokenizer cache, keyed by model path. Tokenizers are stateless
# across requests and much cheaper/safer than loading the full AutoProcessor.
_TOKENIZER_CACHE: dict[str, Any] = {}


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


def _get_tokenizer(model_path: str):
    tok = _TOKENIZER_CACHE.get(model_path)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _TOKENIZER_CACHE[model_path] = tok
    return tok


def _load_pil_images(images: List[Any]):
    """Resolve image inputs (paths/URLs/base64/OpenAI dicts) to PIL.Image (RGB)."""
    from PIL import Image

    from ..mllm import process_image_input

    out = []
    for img in images:
        path = process_image_input(img)
        out.append(Image.open(path).convert("RGB"))
    return out


def _preprocess_images_native(pil_images: List[Any]):
    """Run vMLX's native MiniMax-M3 image preprocessor.

    Returns MLX ``pixel_values`` and ``image_grid_thw`` matching MiniMax's HF
    processor layout, but without importing torch/torchvision.
    """
    import mlx.core as mx

    from .minimax_m3_vl import preprocess_image

    pixel_chunks = []
    grid_chunks = []
    for image in pil_images:
        pv, grid = preprocess_image(image)
        pixel_chunks.append(pv)
        grid_chunks.append(grid)
    if not pixel_chunks:
        raise ValueError("M3 VL: all image inputs failed to preprocess")
    pixel_values = mx.concatenate(pixel_chunks, axis=0).astype(mx.bfloat16)
    grid = mx.concatenate(grid_chunks, axis=0).astype(mx.int32)
    mx.eval(pixel_values, grid)
    return pixel_values, grid


def _expand_image_tokens(text: str, image_grid_thw: Any, merge_size: int = 2) -> str:
    """Expand each MiniMax image marker to one token per merged image patch."""
    image_token = "]<]image[>["
    vision_start = "]<]start of image[>["
    vision_end = "]<]end of image[>["
    placeholder = "]<]placeholder[>["
    merge_length = merge_size**2
    grids = image_grid_thw.tolist()
    index = 0
    while image_token in text:
        if index >= len(grids):
            raise ValueError(
                "M3 VL: prompt contains more image placeholders than images"
            )
        gt, gh, gw = (int(v) for v in grids[index])
        num_tokens = (gt * gh * gw) // merge_length
        text = text.replace(
            image_token,
            vision_start + placeholder * num_tokens + vision_end,
            1,
        )
        index += 1
    if index != len(grids):
        raise ValueError(
            "M3 VL: image count does not match chat-template placeholders"
        )
    return text.replace(placeholder, image_token)


def _tokenize_text(tok: Any, text: str) -> List[int]:
    encoded = tok([text], return_tensors=None)
    ids = encoded["input_ids"][0]
    return [int(x) for x in ids]


def _normalize_messages_for_template(messages: List[dict]) -> Tuple[List[dict], List[Any]]:
    """Return (templated-ready messages, ordered raw image inputs).

    The MiniMax chat template emits an image placeholder for content items of
    the form ``{"type": "image"}``. OpenAI-format requests carry images as
    ``{"type": "image_url", "image_url": {"url": ...}}`` (or a bare string).
    Rewrite those to ``{"type": "image"}`` so the template renders the
    placeholder, and collect the raw image inputs in document order so they line
    up 1:1 with the placeholders.
    """
    out_msgs: List[dict] = []
    raw_images: List[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out_msgs.append(msg)
            continue
        new_content = []
        image_placeholders = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            itype = item.get("type")
            if itype in ("image_url", "image", "input_image"):
                # collect raw source
                src = item.get("image_url", item.get("image", item.get("url")))
                if isinstance(src, dict):
                    src = src.get("url", src)
                if src is not None:
                    raw_images.append(src)
                image_placeholders.append({"type": "image"})
            else:
                new_content.append(item)
        # MiniMax-M3's proven diagnostic path places image placeholders before
        # the text in the same user turn. The panel/OpenAI content-array shape
        # naturally arrives as text then image; preserving that order makes the
        # model behave as if no image was available on mixed text+image turns.
        # Keep raw_images in document order, but render image tokens before the
        # textual prompt for this M3-only preprocessing path.
        out_msgs.append({**msg, "content": image_placeholders + new_content})
    return out_msgs, raw_images


def preprocess_m3_vl_messages(
    model_path: str,
    messages: List[dict],
    *,
    extra_images: Optional[List[Any]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> Optional[Tuple[List[int], Any, Any]]:
    """Template `messages` + native image preprocess -> (input_ids, pv, grid).

    Mirrors MiniMax processor semantics without importing HF AutoProcessor:
    normalize image items to ``{"type": "image"}``, render the chat template,
    preprocess images with the vMLX native MiniMax image path, expand image
    markers to patch-token placeholders, then tokenize.
    """
    tok = _get_tokenizer(model_path)

    norm_msgs, raw_images = _normalize_messages_for_template(messages)
    if not raw_images and extra_images:
        # The server's extract_multimodal_content() flattens message content to
        # plain text and hands the images out-of-band (engine.chat images=...).
        # In that case the templated messages carry no image items, so inject
        # one {"type":"image"} placeholder per extra image into the LAST user
        # turn (matching the diag layout: image(s) precede the text). This is the
        # path exercised by the real /v1/chat/completions server flow.
        raw_images = list(extra_images)
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

    # MiniMax-M3 templates ignore the common enable_thinking kwarg and branch on
    # thinking_mode only. Keep VL preprocessing aligned with server text routes:
    # off -> disabled; on -> enabled; omitted/auto -> adaptive.
    if enable_thinking is False:
        tmpl_kwargs = {"thinking_mode": "disabled"}
    elif enable_thinking is True:
        tmpl_kwargs = {"thinking_mode": "enabled"}
    else:
        tmpl_kwargs = {"thinking_mode": "adaptive"}
    try:
        txt = tok.apply_chat_template(
            norm_msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **tmpl_kwargs,
        )
    except TypeError:
        # Template doesn't accept thinking_mode — retry without it.
        txt = tok.apply_chat_template(
            norm_msgs,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    pil_images = _load_pil_images(raw_images)
    if not pil_images:
        raise ValueError("M3 VL: all image inputs failed to load")

    pixel_values, grid = _preprocess_images_native(pil_images)
    txt = _expand_image_tokens(txt, grid)
    input_ids = _tokenize_text(tok, txt)

    n_img = sum(1 for token in input_ids if token == 200025)
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
    if not images:
        return None

    tok = _get_tokenizer(model_path)
    pil_images = _load_pil_images(images)
    if not pil_images:
        raise ValueError("M3 VL: all image inputs failed to load")

    pixel_values, grid = _preprocess_images_native(pil_images)
    prompt = _expand_image_tokens(prompt, grid)
    input_ids = _tokenize_text(tok, prompt)

    n_img = sum(1 for token in input_ids if token == 200025)
    logger.info(
        "M3 VL preprocess: %d tokens, %d image tokens, pixel_values=%s grid=%s",
        len(input_ids),
        n_img,
        tuple(pixel_values.shape),
        tuple(grid.shape),
    )
    return input_ids, pixel_values, grid

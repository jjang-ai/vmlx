"""DeepSeek-V4.1-Flash vision preprocessing for the text-routed path.

ADDITIVE + GATED. Every entry point here is reached ONLY for a loaded
``deepseek_v41`` bundle that actually carries a vision tower AND a request whose
messages contain image content. Text-only V4.1 (and every other family) never
touches this module, so the proven text + disk-cache serve path is unchanged.

The standalone VL proof (``scratchpad/prove_vl.py``, VL PASS) established the
exact tensors the model perceives:

    prompt, mm = encoding.encode_messages(msgs, thinking_mode=..., reasoning_effort=...,
                                          return_multi_modal_data=True)
    ids  = rt.tokenizer.encode(prompt, add_special_tokens=False).ids
    toks, token_types, images = prepare_vl_inputs(ids, mm["images"], rt.vision_args)
    model(mx.array(toks)[None], cache=cache, images=images, token_types=token_types)

This module reproduces that preprocessing inside the engine so the server feeds
the SingleBatchGenerator identical ``input_ids`` / ``images`` / ``token_types``.
The SingleBatchGenerator splices each image into the prefill chunk that holds
its span (chunk-relative start; chunks never split an image), so images may sit
anywhere in a long multi-turn prompt.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}

# Process-wide cache of each bundle's encoding/encoding.py module. The module is
# stateless across requests; loading it (exec of the bundle file) is not free.
_ENCODING_CACHE: dict[str, Any] = {}


def dsv41_vl_enabled() -> bool:
    """True unless explicitly disabled. The real gate is structural detection
    (:func:`is_dsv41_vl_model`); this env switch only lets an operator force
    text-only on a VL bundle via ``VMLX_DSV41_VL=0``."""
    return os.environ.get("VMLX_DSV41_VL", "").strip().lower() not in _FALSE


def _runtime(model: Any) -> Optional[Any]:
    """Resolve the attached ``DSV41`` runtime through engine loader wrappers."""
    for getter in (
        lambda: getattr(model, "_rt", None),
        lambda: getattr(getattr(model, "model", None), "_rt", None),
    ):
        try:
            rt = getter()
        except Exception:
            rt = None
        if rt is not None:
            return rt
    return None


def is_dsv41_vl_model(model: Any) -> bool:
    """True iff ``model`` is a deepseek_v41 build carrying a vision tower.

    Structural (model_type == deepseek_v41 AND the runtime exposes a non-None
    ``vision_args``), robust to loader wrappers. Reads ``_rt`` directly rather
    than the ``runtime`` property so a text-only wrapper never raises.
    """
    try:
        mt = str(getattr(model, "model_type", "")).lower()
        if mt != "deepseek_v41":
            inner = getattr(model, "model", None)
            mt = str(getattr(inner, "model_type", "")).lower()
        if mt != "deepseek_v41":
            return False
        rt = _runtime(model)
        return rt is not None and getattr(rt, "vision_args", None) is not None
    except Exception:
        return False


def messages_have_image(messages: List[dict]) -> bool:
    """True iff any message content part carries an image (OpenAI shapes)."""
    for msg in messages or ():
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type", "") or "").lower()
            if itype in ("image_url", "image", "input_image"):
                return True
            if any(k in item for k in ("image", "image_url")):
                return True
    return False


def _encoding(model_path: str):
    mod = _ENCODING_CACHE.get(model_path)
    if mod is not None:
        return mod
    p = Path(model_path) / "encoding" / "encoding.py"
    if not p.exists():
        raise FileNotFoundError(f"deepseek_v41 VL: no encoding module at {p}")
    spec = importlib.util.spec_from_file_location(
        f"dsv41_encoding_{abs(hash(model_path))}", p
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _ENCODING_CACHE[model_path] = mod
    return mod


def _normalize_messages_for_encoding(messages: List[dict]) -> List[dict]:
    """Rewrite OpenAI image parts to the bundle encoder's ``image_url`` shape.

    The bundle ``encode_messages`` consumes ``{"type":"image_url","image_url":
    {"url": <data/url>}}`` (the shape ``prove_vl.py`` used). Panel/other clients
    may send ``{"type":"image","image":<url>}`` or ``{"type":"input_image",...}``;
    normalize those so no image is silently dropped. Text parts pass through.
    """
    out: List[dict] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_content = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            itype = str(item.get("type", "") or "").lower()
            if itype in ("image_url", "image", "input_image"):
                src = item.get("image_url", item.get("image", item.get("url")))
                if isinstance(src, dict):
                    url = src.get("url", src)
                else:
                    url = src
                if url is not None:
                    new_content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                new_content.append(item)
        out.append({**msg, "content": new_content})
    return out


def _inject_extra_images(messages: List[dict], extra_images: List[Any]) -> List[dict]:
    """Inject out-of-band images as image_url parts into the LAST user turn.

    The server's text-route strips image parts from ``messages`` and hands the
    images separately (``extract_multimodal_content``). Rebuild the image parts
    (image(s) before the text, matching the proven VL layout) so the bundle
    encoder renders image placeholders and returns the loaded images.
    """
    placeholders = [{"type": "image_url", "image_url": {"url": img}} for img in extra_images]
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            c = out[i].get("content")
            if isinstance(c, str):
                new_c = placeholders + [{"type": "text", "text": c}]
            elif isinstance(c, list):
                new_c = placeholders + list(c)
            else:
                new_c = placeholders
            out[i] = {**out[i], "content": new_c}
            return out
    return out + [{"role": "user", "content": placeholders}]


def _image_parts(content: Any) -> List[dict]:
    if not isinstance(content, list):
        return []
    return [
        p for p in _normalize_messages_for_encoding([{"role": "user", "content": content}])[0]["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]


def _is_user_turn(msg: Any) -> bool:
    # Tool results rendered as user-shaped turns carry this marker (api.utils).
    return isinstance(msg, dict) and msg.get("role") == "user" and msg.get("type") != "function_call_output"


def anchor_images_to_turns(
    messages: List[dict], source_messages: Optional[List[dict]]
) -> Optional[Tuple[List[dict], int]]:
    """Put each client user turn's image parts back into the matching user turn of ``messages``.

    The server's text route strips image parts (``extract_multimodal_content``) and the old code
    re-injected every image into the LAST user turn, so a conversation with an image in turn 1 was
    re-rendered as if the image had been sent with the latest question. ``source_messages`` are the
    client messages before stripping; user turns are matched by ordinal (normalization never adds,
    drops or reorders user turns). When the stripped text equals the source's text parts joined
    with "\n" (no rewrite happened) the source's own part order is restored; otherwise the images
    go first, then the (rewritten) text -- in the right turn either way. Returns (messages, n_placed)
    or None when the turns cannot be matched."""
    if not source_messages:
        return None
    src = [m for m in source_messages if isinstance(m, dict) and m.get("role") == "user"]
    dst = [i for i, m in enumerate(messages) if _is_user_turn(m)]
    if len(src) != len(dst):
        return None
    out = [dict(m) for m in messages]
    placed = 0
    for sm, i in zip(src, dst):
        content = sm.get("content")
        imgs = _image_parts(content)
        if not imgs:
            continue
        ordered = _normalize_messages_for_encoding([{"role": "user", "content": content}])[0]["content"]
        texts = [p.get("text", "") for p in ordered if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
        cur = out[i].get("content")
        if isinstance(cur, str) and cur == "\n".join(texts):
            out[i]["content"] = [
                p if p.get("type") != "input_text" else {"type": "text", "text": p.get("text", "")}
                for p in ordered
                if isinstance(p, dict) and p.get("type") in ("text", "input_text", "image_url")
            ]
        else:
            rest = cur if isinstance(cur, list) else ([{"type": "text", "text": cur}] if cur else [])
            out[i]["content"] = imgs + rest
        placed += len(imgs)
    return out, placed


def _attach_tools(messages: List[dict], tools: Optional[list]) -> List[dict]:
    """Same tool placement as the text path's apply_chat_template shim (loaders/load_jangtq_dsv41):
    the encoder renders tools only from a leading system message."""
    if not tools:
        return messages
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["tools"] = tools
    else:
        msgs = [{"role": "system", "content": "", "tools": tools}] + msgs
    return msgs


def media_cache_extra_keys(images: List[Any]) -> Optional[dict]:
    """Prefix-cache salt for a VL request: SHA-256 over every image's pixels + geometry, in prompt order.

    Image spans are runs of the same placeholder token id, so token ids alone cannot tell two images
    apart; the salt keeps vision-conditioned state from being reused for different pixels while letting
    the same conversation (same images, more turns) hit its own earlier prompt boundary."""
    if not images:
        return None
    import hashlib

    import mlx.core as mx
    import numpy as np

    h = hashlib.sha256(b"vmlx-dsv41-vl-media-v1")
    for img in images:
        patches = getattr(img, "patches", None)
        h.update(repr((int(img.n_vit_h), int(img.n_vit_w), len(img.types))).encode())
        h.update(np.asarray(img.types, np.int16).tobytes())
        if patches is not None:
            arr = patches
            if arr.dtype in (mx.bfloat16, mx.float16):
                arr = arr.view(mx.uint16)
            h.update(np.asarray(arr).tobytes())
    return {"dsv41_vl_media": h.hexdigest()[:32]}


def preprocess_dsv41_vl(
    model: Any,
    model_path: str,
    messages: List[dict],
    *,
    extra_images: Optional[List[Any]] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Any = None,
    tools: Optional[list] = None,
    source_messages: Optional[List[dict]] = None,
) -> Optional[Tuple[List[int], List[Any], List[int]]]:
    """Return ``(input_ids, images, token_types)`` for a V4.1 VL request.

    ``input_ids`` already have image spans expanded; ``images`` is the
    ``jang_tools.dsv41.vision.ImageInput`` list; ``token_types`` is per-token
    (>=0 on image-span positions). ``extra_images`` carries the server's
    out-of-band image list (``messages`` text-route strips image parts); it is
    injected back into the last user turn. Returns ``None`` when the model is
    not a VL build or no image is present (caller falls back to the text path).
    Raises on a genuine preprocessing failure (never silently drops an image).
    """
    rt = _runtime(model)
    if rt is None or getattr(rt, "vision_args", None) is None:
        return None

    from jang_tools.dsv41.vision import prepare_vl_inputs

    enc = _encoding(model_path)
    thinking_mode = "thinking" if enable_thinking else "chat"
    if messages_have_image(messages):
        norm_msgs = _normalize_messages_for_encoding(messages)
    elif extra_images:
        anchored = anchor_images_to_turns(messages, source_messages)
        if anchored is not None and anchored[1] == len(extra_images):
            norm_msgs = _normalize_messages_for_encoding(anchored[0])
        else:
            if source_messages:
                logger.warning(
                    "DSV4.1 VL: could not re-anchor %d image(s) to their original turns; "
                    "placing them in the last user turn",
                    len(extra_images),
                )
            norm_msgs = _normalize_messages_for_encoding(
                _inject_extra_images(messages, list(extra_images))
            )
    else:
        return None
    # The text path renders through the loader's apply_chat_template shim, whose reasoning_effort
    # default is "high"; render image turns identically (and with the same tools) so an image turn
    # and its text continuation share one prompt format (and a prefix-cache boundary).
    prompt, mm = enc.encode_messages(
        _attach_tools(norm_msgs, tools),
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort if reasoning_effort is not None else "high",
        return_multi_modal_data=True,
    )
    raw_images = (mm or {}).get("images") or []
    if not raw_images:
        return None
    ids = rt.tokenizer.encode(prompt, add_special_tokens=False).ids
    toks, token_types, images = prepare_vl_inputs(ids, raw_images, rt.vision_args)

    # Materialize the pixel tensors NOW on this (server event-loop) thread so no
    # lazy graph crosses into the scheduler worker thread, where resolving an
    # array bound to a stale default stream raises "no Stream(gpu,0) in current
    # thread" (the P0 VL stream bug M3 documents). After eval they are concrete
    # data the worker thread just reads.
    import mlx.core as mx

    _patches = [im.patches for im in images if getattr(im, "patches", None) is not None]
    if _patches:
        mx.eval(*_patches)

    n_span = int(sum(1 for t in token_types if t >= 0))
    logger.info(
        "DSV4.1 VL preprocess: %d tokens, %d image-span tokens, %d image(s)",
        len(toks),
        n_span,
        len(images),
    )
    if n_span == 0:
        raise ValueError(
            "deepseek_v41 VL: encoder produced no image-span tokens; refusing to "
            "silently drop the image."
        )
    return list(toks), list(images), list(token_types)

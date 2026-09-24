"""Reconstruct a native Omni prompt from the complete client transcript.

This path is used after a session miss. Earlier assistant replies are teacher
forced as prompt text, never regenerated. Encoders remain the native session's
reference encoders, and only placeholder embeddings replace text embeddings.
"""
from __future__ import annotations

from copy import deepcopy


def run_full_history(session, messages, *, scratch_dir, extract_parts,
                     enable_thinking, max_tokens, temperature, top_p,
                     token_callback=None):
    import numpy as np
    from PIL import Image

    rendered = []
    visual_groups = []
    audio_groups = []
    for message in messages:
        item = deepcopy(message)
        content = item.get("content")
        if isinstance(content, list):
            texts = []
            visual_tokens = 0
            audio_tokens = 0
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text") or "")
                    continue
                _, images, audio, video = extract_parts(
                    [{"role": "user", "content": [part]}], scratch_dir,
                )
                if not images and audio is None and video is None:
                    raise ValueError("Omni media part could not be decoded")
                if images:
                    pil_images = []
                    try:
                        for path in images:
                            with Image.open(path) as source:
                                pil_images.append(source.convert("RGB"))
                        embeds = session._extract_image_embeddings(pil_images)
                    finally:
                        for image in pil_images:
                            image.close()
                    flat = embeds.reshape(-1, embeds.shape[-1])
                    visual_groups.append(flat)
                    visual_tokens += len(flat)
                if video is not None:
                    embeds = session._extract_video_embeddings(str(video))
                    flat = embeds.reshape(-1, embeds.shape[-1])
                    visual_groups.append(flat)
                    visual_tokens += len(flat)
                if audio is not None:
                    embeds = session._extract_audio_embeddings(str(audio))
                    flat = embeds.reshape(-1, embeds.shape[-1])
                    audio_groups.append(flat)
                    audio_tokens += len(flat)
            # Preserve the reference session's media-first turn layout. Across
            # turns, every embedding remains at its original causal position.
            media = ""
            if visual_tokens:
                media += "<img>" + "<image>" * visual_tokens + "</img>\n"
            if audio_tokens:
                media += "<sound>" + "<so_embedding>" * audio_tokens + "</sound>\n"
            item["content"] = media + "".join(texts)
        rendered.append(item)

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    prompt = session.tokenizer.apply_chat_template(rendered, **kwargs)
    input_ids = session.tokenizer(prompt, return_tensors="np")["input_ids"]
    session._ensure_cache()
    session._last_prompt_tokens = int(input_ids.shape[-1])
    embeds = session.mlx_model.backbone.embeddings(session.mx.array(input_ids))
    visuals = np.concatenate(visual_groups, axis=0)[None, ...] if visual_groups else None
    audio = np.concatenate(audio_groups, axis=0)[None, ...] if audio_groups else None
    embeds = session._inject_embeddings(input_ids, embeds, visuals, None, audio)
    reply = session._decode_turn(
        embeds, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        token_callback=token_callback,
    )
    session._history_text = rendered + [{"role": "assistant", "content": reply}]
    return reply

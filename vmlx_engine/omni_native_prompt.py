"""Reconstruct a native Omni prompt from the complete client transcript.

This path is used after a session miss. Earlier assistant replies are teacher
forced as prompt text, never regenerated. Encoders remain the native session's
reference encoders, and only placeholder embeddings replace text embeddings.
"""
from __future__ import annotations

from copy import deepcopy


def run_full_history(session, messages, *, scratch_dir, extract_parts,
                     enable_thinking, max_tokens, temperature, top_p,
                     token_callback=None, checkpoints=None, template_options=None,
                     tools=None):
    import numpy as np
    from PIL import Image

    candidate = checkpoints.find() if checkpoints is not None else None
    source_count = candidate["source_count"] if candidate else 0
    rendered = deepcopy(candidate["rendered"]) if candidate else []
    visual_groups = []
    audio_groups = []
    for message in messages[source_count:]:
        item = deepcopy(message)
        content = item.get("content")
        if isinstance(content, list):
            texts = []
            visual_tokens = 0
            audio_tokens = 0
            visual_fragments = []
            has_temporal_video = False
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
                    visual_fragments.append("<img>" + "<image>" * len(flat) + "</img>\n")
                if video is not None:
                    embeds = session._extract_video_embeddings(str(video))
                    flat = embeds.reshape(-1, embeds.shape[-1])
                    visual_groups.append(flat)
                    visual_tokens += len(flat)
                    video_prompt = getattr(session, "_vmlx_video_prompt", None)
                    if video_prompt is not None:
                        if video_prompt.count("<image>") != len(flat):
                            raise ValueError("Native video prompt and embedding token counts disagree")
                        has_temporal_video = True
                    visual_fragments.append(video_prompt or "<img>" + "<image>" * len(flat) + "</img>\n")
                if audio is not None:
                    embeds = session._extract_audio_embeddings(str(audio))
                    flat = embeds.reshape(-1, embeds.shape[-1])
                    audio_groups.append(flat)
                    audio_tokens += len(flat)
            # Preserve the reference session's media-first turn layout. Across
            # turns, every embedding remains at its original causal position.
            media = ""
            if has_temporal_video:
                media += "".join(visual_fragments)
            elif visual_tokens:
                media += "<img>" + "<image>" * visual_tokens + "</img>\n"
            if audio_tokens:
                media += "<sound>" + "<so_embedding>" * audio_tokens + "</sound>\n"
            item["content"] = media + "".join(texts)
        rendered.append(item)

    kwargs = {**(template_options or {}), "tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    prompt = session.tokenizer.apply_chat_template(rendered, **kwargs)
    input_ids = session.tokenizer(prompt, return_tensors="np")["input_ids"]
    restored = checkpoints.accept(input_ids) if checkpoints is not None else 0
    if restored is None:
        # A native template may revise earlier turns (e.g. move a budget hint).
        # Re-encode the supplied media instead of using incompatible RNN state.
        return run_full_history(
            session, messages, scratch_dir=scratch_dir, extract_parts=extract_parts,
            enable_thinking=enable_thinking, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, token_callback=token_callback,
            checkpoints=checkpoints, template_options=template_options, tools=tools,
        )
    if hasattr(session, "_vmlx_prompt_rail_callback"):
        from .omni_native_controls import record_prompt_rail
        record_prompt_rail(session, prompt)
    session._ensure_cache()
    session._vmlx_restored_prefix_tokens = restored
    session._last_prompt_tokens = int(input_ids.shape[-1]) - restored
    suffix_ids = input_ids[:, restored:]
    embeds = session.mlx_model.backbone.embeddings(session.mx.array(suffix_ids))
    visuals = np.concatenate(visual_groups, axis=0)[None, ...] if visual_groups else None
    audio = np.concatenate(audio_groups, axis=0)[None, ...] if audio_groups else None
    embeds = session._inject_embeddings(suffix_ids, embeds, visuals, None, audio)
    if checkpoints is not None and checkpoints.publish_enabled:
        boundary_prompt = session.tokenizer.apply_chat_template(
            rendered, **{**kwargs, "add_generation_prompt": False},
        )
        boundary_ids = session.tokenizer(boundary_prompt, return_tensors="np")["input_ids"]
        boundary = int(boundary_ids.shape[-1])
        if (restored <= boundary < input_ids.shape[-1]
                and input_ids[0, :boundary].tolist() == boundary_ids[0].tolist()):
            from .omni_native_prefix import prefill_state
            prefill_state(session, embeds[:, :boundary - restored])
            checkpoints.publish(boundary_ids, rendered)
            embeds = embeds[:, boundary - restored:]
    reply = session._decode_turn(
        embeds, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        token_callback=token_callback,
    )
    session._history_text = rendered + [{"role": "assistant", "content": reply}]
    return reply

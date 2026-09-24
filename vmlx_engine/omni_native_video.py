"""Native temporal RADIO video encoding for bundles carrying trained weights."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_PREFIX = "vision_model.radio_model.model.patch_generator."


def temporal_video_spec(bundle_path):
    """Cross-check indexed temporal/image projections against shard headers."""
    from .model_bundle_integrity import _read_safetensors_header

    root = Path(bundle_path).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text())["weight_map"]
    video_key, image_key = _PREFIX + "video_embedder.weight", _PREFIX + "embedder.weight"
    if video_key not in index:
        return None
    headers = {}
    tensors = {}
    for key in (video_key, image_key):
        if key not in index:
            raise ValueError(f"Native Omni video is missing its image projection: {key}")
        path = (root / index[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native Omni projection shard escapes its bundle")
        if path not in headers:
            headers[path] = _read_safetensors_header(path)["tensors"]
        tensor = headers[path].get(key)
        if not tensor or tensor.get("dtype") not in {"F16", "BF16", "F32"}:
            raise ValueError(f"Native Omni projection is missing or not floating point: {key}")
        shape = tensor.get("shape", [])
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"Native Omni projection has an invalid shape: {key} {shape}")
        tensors[key] = tensor
    video, image = tensors[video_key]["shape"], tensors[image_key]["shape"]
    if video[0] != image[0] or video[1] % image[1]:
        raise ValueError("Native Omni temporal and image projection shapes disagree")
    temporal = video[1] // image[1]
    if temporal < 1:
        raise ValueError("Native Omni temporal patch size must be positive")
    config = json.loads((root / "config_omni.json").read_text())
    declared = config.get("video_temporal_patch_size")
    if declared is not None and declared != temporal:
        raise ValueError("Native Omni temporal patch metadata disagrees with shard weights")
    return {"temporal_patch_size": temporal, "weight": video_key,
            "shape": video, "dtype": tensors[video_key]["dtype"]}


def encode_temporal_video(session, video_path, *, controls, temporal_patch_size):
    """Use native video resizing and trained temporal projection, without sheets.

    The native model pads the final temporal group internally when required.
    The frame ceiling counts actual sampled frames, not that internal padding.
    """
    import numpy as np
    import torch
    from PIL import Image
    from .models.mllm import extract_video_frames_smart
    from .video_controls import subsample_frames_evenly

    if session.pt_model.video_temporal_patch_dim != temporal_patch_size:
        raise ValueError("Loaded native video temporal size disagrees with shard weights")
    frames = extract_video_frames_smart(
        str(video_path), fps=controls["fps"], max_frames=controls["max_frames"],
    )
    frames = subsample_frames_evenly(frames, min(len(frames), controls["max_frames"]))
    if not frames:
        raise ValueError("video contains no readable frames")
    # Retain repeated frames: dropping them changes temporal grouping/motion.
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    processor = session.processor.image_processor
    previous_mode = getattr(processor, "_is_video_mode", False)
    try:
        processor._is_video_mode = True
        processed = processor(images=images, return_tensors="pt")
    finally:
        processor._is_video_mode = previous_mode
        for image in images:
            image.close()
    pixels = processed["pixel_values"].to(session.device, dtype=session.torch_dtype)
    with torch.no_grad():
        features = session.pt_model.extract_video_feature(pixels)
    embeds = features.detach().to("cpu", dtype=torch.float32).numpy()
    groups = (len(frames) + temporal_patch_size - 1) // temporal_patch_size
    if embeds.ndim != 3 or embeds.shape[0] != groups:
        raise ValueError("Native video output does not match sampled temporal groups")
    # Match the bundle processor's no-timestamp form. Never invent timestamps
    # when the shared frame sampler does not return source frame indices.
    fragments = []
    for group in range(groups):
        labels = [
            f"{'Frame' if offset == 0 else 'frame'} {group * temporal_patch_size + offset + 1}"
            for offset in range(temporal_patch_size)
            if group * temporal_patch_size + offset < len(frames)
        ]
        fragments.append(" and ".join(labels) + ": <img>" +
                         "<image>" * embeds.shape[1] + "</img>")
    prompt = "\n".join(fragments) + "\n"
    logger.info("Omni native video: fps=%s frame_cap=%s sampled=%d temporal_patch=%d groups=%d tokens=%d",
                controls["fps"], controls["max_frames"], len(frames), temporal_patch_size,
                groups, int(np.prod(embeds.shape[:2])))
    return embeds, prompt

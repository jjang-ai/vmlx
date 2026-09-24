"""Read native Omni media component evidence without materializing weights."""
from __future__ import annotations

from pathlib import Path
from .model_bundle_integrity import _read_safetensors_header

_MEDIA_PREFIXES = ("vision_model.", "sound_encoder.", "parakeet.",
                   "mlp1.", "sound_projection.", "sound_projector.", "projector.")
_FLOAT = {"F16", "BF16", "F32"}


def inspect_media_weights(root, weight_map, config, omni_config):
    """Cross-check every indexed media entry in its actual shard.

    This is component admission, not proof that every encoder layer or modality
    works. Runtime qualification still requires loading the exact artifact.
    """
    root = Path(root)
    if not isinstance(weight_map, dict):
        raise ValueError("native media weight_map must be an object")
    headers, tensors = {}, {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name.startswith(_MEDIA_PREFIXES):
            continue
        if (not isinstance(shard, str) or Path(shard).is_absolute()
                or ".." in Path(shard).parts):
            raise ValueError("native media shard path is invalid")
        path = root / shard
        if path not in headers:
            headers[path] = _read_safetensors_header(path)["tensors"]
        tensor = headers[path].get(name)
        if tensor is None:
            raise ValueError(f"indexed native media tensor is missing: {name}")
        if ((name.endswith(".weight") and not tensor["shape"])
                or any(d <= 0 for d in tensor["shape"])):
            raise ValueError(f"native media tensor has an empty shape: {name}")
        if name.endswith(".weight") and tensor["dtype"] not in _FLOAT:
            raise ValueError(f"native media weight is not supported floating point: {name}")
        tensors[name] = tensor

    def shape(name, rank):
        tensor = tensors.get(name)
        if not tensor or tensor["dtype"] not in _FLOAT or len(tensor["shape"]) != rank:
            return None
        return tensor["shape"]

    text_hidden = config.get("hidden_size")
    omni_hidden = (omni_config.get("llm_config") or {}).get("hidden_size")
    if text_hidden is not None and omni_hidden is not None and text_hidden != omni_hidden:
        raise ValueError("native media and text hidden sizes disagree")
    hidden = text_hidden or omni_hidden
    image = shape("vision_model.radio_model.model.patch_generator.embedder.weight", 2)
    vision_norm = shape("mlp1.0.weight", 1)
    vision_in = shape("mlp1.1.weight", 2)
    vision_out = shape("mlp1.3.weight", 2)
    vision_projector = bool(vision_norm and vision_in and vision_out
                            and vision_norm[0] == vision_in[1]
                            and vision_in[0] == vision_out[1]
                            and (hidden is None or vision_out[0] == hidden)
                            and (omni_config.get("projector_hidden_size") is None
                                 or vision_in[0] == omni_config["projector_hidden_size"]))
    downsample = omni_config.get("downsample_ratio")
    if image and vision_projector and downsample is not None:
        try:
            factor = 1 / float(downsample)
            vision_projector = factor >= 1 and factor.is_integer() and vision_in[1] == image[0] * int(factor) ** 2
        except (TypeError, ValueError, ZeroDivisionError):
            vision_projector = False

    sound_norm = shape("sound_projection.norm.weight", 1)
    sound_in = shape("sound_projection.linear1.weight", 2)
    sound_out = shape("sound_projection.linear2.weight", 2)
    sound_config = omni_config.get("sound_config") or {}
    sound_hidden = sound_config.get("hidden_size")
    audio_projector = bool(sound_norm and sound_in and sound_out
                           and sound_norm[0] == sound_in[1]
                           and sound_in[0] == sound_out[1]
                           and (hidden is None or sound_out[0] == hidden)
                           and (sound_hidden is None or sound_norm[0] == sound_hidden)
                           and (sound_config.get("projection_hidden_size") is None
                                or sound_in[0] == sound_config["projection_hidden_size"]))
    radio = bool(image and any(name.startswith("vision_model.radio_model.model.blocks.")
                              for name in tensors))
    parakeet = any(name.startswith(("sound_encoder.", "parakeet.")) for name in tensors)
    return {
        "has_radio_weights": radio,
        "has_parakeet_weights": parakeet,
        "has_vision_projector": vision_projector,
        "has_audio_projector": audio_projector,
        "has_media_projector": vision_projector and audio_projector,
        "weight_evidence": {
            "indexed_media_tensors_verified": len(tensors),
            "shards": sorted(str(path.relative_to(root)) for path in headers),
            "vision_output_shape": vision_out,
            "audio_output_shape": sound_out,
            "method": "indexed_tensor_headers_and_projection_shapes",
        },
    }

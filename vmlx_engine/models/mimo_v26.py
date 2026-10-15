"""Fresh MiMo-V2.6 bridge for vMLX's standard multimodal scheduler.

Text loading uses the measured mixed affine/MXFP4 runtime. The processors and
towers come from the V2.6 reference ports; no legacy MiMo VLM registration or
global SwitchGLU patch is used. Towers remain lazy for text-only requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput

from jang_tools.mimo_v2 import v26_audio as audio_port
from jang_tools.mimo_v2 import v26_vision as vision_port
from jang_tools.mimo_v2 import v26_model as text_port
from jang_tools.mimo_v2 import v26_omni as omni_port
from jang_tools.mimo_v2 import mlx_register as registration_port
from jang_tools.mimo_v2.v26_omni import MiMoV26Omni
from .mimo_v26_contract import (
    canonicalize_mimo_v26_tool_results,
    mimo_v26_cache_identity,
    mimo_v26_runtime_source_identity,
)


# These modules live outside the engine source tree. Their distribution version
# can stay unchanged during development, so the engine fingerprint alone is
# insufficient. Freeze at import, like the shared runtime cache fingerprint.
_MIMO_RUNTIME_IDENTITY = mimo_v26_runtime_source_identity({
    module.__name__: Path(module.__file__)
    for module in (audio_port, vision_port, text_port, omni_port, registration_port)
})


class LanguageModel(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.args = inner.args
        self.model_type = inner.model_type

    @property
    def model(self):
        return self.inner.model

    @property
    def layers(self):
        return self.inner.layers

    def make_cache(self):
        return self.inner.make_cache()

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        return LanguageModelOutput(logits=self.inner(
            inputs, cache=cache, input_embeddings=inputs_embeds,
        ))


class Model(nn.Module):
    _mimo_v26_runtime = True

    def __init__(self, text, omni, config):
        super().__init__()
        self.language_model = LanguageModel(text)
        self.omni = omni
        self.config = SimpleNamespace(**config)
        self.model_type = "mimo_v2"

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(self, input_ids, pixel_values=None,
                             image_grid_thw=None, audio_codes=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("MiMo-V2.6 media requires one request per prefill")
        emb = self.language_model.model.embed_tokens(input_ids)
        ids = np.array(input_ids[0])
        if pixel_values is not None:
            tower, _ = self.omni.vision()
            grid = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
            features = tower(pixel_values.astype(self.omni.vision_dtype), grid)
            positions = np.flatnonzero(np.isin(ids, [vision_port.IMAGE_PAD_ID, vision_port.VIDEO_PAD_ID]))
            if features.shape[0] != len(positions):
                raise ValueError("MiMo visual feature count differs from expanded placeholders")
            emb[0, mx.array(positions)] = features.astype(emb.dtype)
        if audio_codes is not None:
            _, encoder = self.omni.audio()
            features = audio_port.embed_audio(audio_codes, encoder)
            positions = np.flatnonzero(ids == audio_port.AUDIO_TOKEN_ID)
            if features.shape[0] != len(positions):
                raise ValueError("MiMo audio feature count differs from expanded placeholders")
            emb[0, mx.array(positions)] = features.astype(emb.dtype)
        return InputEmbeddingsFeatures(inputs_embeds=emb)

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values=pixel_values, **kwargs)
        return self.language_model(input_ids, cache=cache, inputs_embeds=features.inputs_embeds)


class MiMoV26Processor:
    model_type = "mimo_v2"
    _mimo_v26_runtime = True
    supports_video_timestamps = True

    def __init__(self, tokenizer, omni):
        self.tokenizer, self.omni = tokenizer, omni
        self.chat_template = tokenizer.chat_template
        self.video_processor = SimpleNamespace(fps=1.0)

    def __getattr__(self, name):
        # Do not expose a non-callable tokenizer wrapper as `.process`.
        if name == "process":
            raise AttributeError(name)
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(
            canonicalize_mimo_v26_tool_results(messages), **kwargs,
        )

    @staticmethod
    def _audio(value):
        if isinstance(value, tuple) and len(value) == 2:
            return np.asarray(value[0]), int(value[1])
        if isinstance(value, (str, Path)):
            import soundfile as sf
            wav, sr = sf.read(str(value), dtype="float32", always_2d=False)
            return wav.T if wav.ndim == 2 else wav, sr
        raise ValueError("MiMo audio requires a decoded file or (waveform, sample_rate)")

    def __call__(self, text, images=None, videos=None, audio=None,
                 fps=None, video_timestamps=None, add_special_tokens=False,
                 padding=True, return_tensors="mlx", **kwargs):
        if isinstance(text, list):
            if len(text) != 1:
                raise ValueError("MiMo media processor accepts one prompt at a time")
            text = text[0]
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        items = {"image": list(images or []), "video": list(videos or []), "audio": list(audio or [])}
        triples = {
            "image": (151652, 151655, 151653),
            "video": (151652, 151656, 151653),
            "audio": (151673, 151669, 151674),
        }
        spans = sorted((p, kind) for kind, triple in triples.items()
                       for p in MiMoV26Omni._find(ids, triple))
        for kind, values in items.items():
            if len(values) != sum(k == kind for _, k in spans):
                raise ValueError(f"MiMo {kind} attachment/placeholder count mismatch")
        cursor, output, visual, codes = 0, [], [], []
        media_items, pixel_rows, code_rows = [], 0, 0
        offsets = {kind: 0 for kind in items}
        for position, kind in spans:
            output.extend(ids[cursor:position])
            index = offsets[kind]
            value = items[kind][index]
            offsets[kind] += 1
            token_begin = len(output)
            record = {"modality": kind, "source_index": index}
            if kind == "audio":
                tokenizer, _ = self.omni.audio()
                wav, sr = self._audio(value)
                code, count = audio_port.prepare_audio(wav, sr, tokenizer)
                # Each clip pads independently before four-frame grouping.
                pad = (-code.shape[0]) % 4
                if pad:
                    code = mx.concatenate([code, mx.repeat(code[-1:], pad, axis=0)])
                record.update(code_start=code_rows, code_end=code_rows+int(code.shape[0]))
                code_rows += int(code.shape[0])
                codes.append(code)
                output.extend(audio_port.build_audio_prompt_ids(count))
            else:
                _, processor = self.omni.vision()
                if kind == "image":
                    if isinstance(value, (str, Path)):
                        from mlx_vlm.utils import load_image
                        value = load_image(value)
                    item = processor.preprocess_image(value)
                else:
                    if isinstance(value, (str, Path)):
                        from mlx_vlm.utils import load_video
                        value, sample_fps = load_video(str(value), fps=1.0)
                    else:
                        sample_fps = fps[index] if isinstance(fps, list) else (fps or 1.0)
                    frames = np.asarray(value)
                    if frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[-1] != 3:
                        frames = frames.transpose(0, 2, 3, 1)
                    timestamps = (video_timestamps[index] if video_timestamps is not None
                                  else np.arange(len(frames)) / sample_fps)
                    item = processor.preprocess_video(frames, timestamps)
                record.update(pixel_start=pixel_rows, pixel_end=pixel_rows+int(item.pixel_values.shape[0]))
                pixel_rows += int(item.pixel_values.shape[0])
                visual.append(item)
                output.extend(processor.expand_tokens(item, encode=lambda s: self.tokenizer.encode(s, add_special_tokens=False)))
            pads = [j for j in range(token_begin, len(output)) if output[j] == triples[kind][1]]
            record.update(token_start=pads[0], token_end=pads[-1]+1, pad_tokens=len(pads))
            media_items.append(record)
            cursor = position + 3
        output.extend(ids[cursor:])
        result = {"input_ids": mx.array([output], dtype=mx.int32),
                  "attention_mask": mx.ones((1, len(output)), dtype=mx.int32)}
        if media_items:
            result["_vmlx_mimo26_media_items"] = media_items
        if visual:
            pixels, grid = vision_port.batch_items(visual)
            result.update(pixel_values=mx.array(pixels), image_grid_thw=mx.array(grid))
        if codes:
            result["audio_codes"] = mx.concatenate(codes)
        return result


def load_mimo_v26(bundle):
    from jang_tools.mimo_v2.mlx_register import register
    from mlx_lm import load

    register()
    from .mimo_v26_contract import read_mimo_v26_contract
    from .mimo_v26_janght import is_janght_bundle, load_mimo_v26_janght_text
    if is_janght_bundle(read_mimo_v26_contract(bundle)):
        text, tokenizer = load_mimo_v26_janght_text(bundle)
    else:
        text, tokenizer = load(str(bundle))
    config = json.loads((Path(bundle) / "config.json").read_text())
    omni = MiMoV26Omni(bundle, text, tokenizer)
    model = Model(text, omni, config)
    identity = mimo_v26_cache_identity(bundle, _MIMO_RUNTIME_IDENTITY, {
        "vision_dtype": str(omni.vision_dtype),
        "audio_tokenizer_dtype": str(omni.audio_tokenizer_dtype),
        "merger_norm": omni.merger_norm,
    })
    for owner in (model, model.language_model, text):
        owner._vmlx_runtime_artifact_identity = identity
    materialized = getattr(text, "_vmlx_prefill_scores_materialized", True)
    for owner in (model, model.language_model):
        owner._vmlx_prefill_scores_materialized = materialized
    return model, MiMoV26Processor(tokenizer, omni), config

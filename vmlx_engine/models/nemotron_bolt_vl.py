"""Nemotron-3.5-Bolt (nemotron_h puzzle LLM + C-RADIO v4-H vision) on vMLX's standard multimodal scheduler.

Text loading is the ordinary JANG path (``load_jang_model``: per-block MoE patch, JANGH relu2 experts, affine
non-experts). Images go through ``jang_tools.nemotron_bolt.vision.BoltVision`` (native MLX tower + relu2 projector,
parity with NVIDIA's torch tower on Bolt's own weights) and are spliced into the token embeddings at the ``<image>``
(img_context_token_id) positions — the same rows the JANG calibration and held-out evaluation used. The bundle's own
processor does placeholder expansion and dynamic resolution. The tower loads lazily, so text-only requests never touch
it. Video is not wired yet (requests with video parts fail closed).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput

logger = logging.getLogger(__name__)


def is_nemotron_bolt_vl_bundle(path) -> bool:
    p = Path(path)
    try:
        cfg = json.loads((p / "config.json").read_text())
        omni = json.loads((p / "config_omni.json").read_text())
    except (OSError, ValueError):
        return False
    vc = omni.get("vision_config") or {}
    # Bolt (puzzle, block_configs) and Nemotron-3.5-Super-VL (uniform MoE + MTP-implied vision final LayerNorm, handled
    # by BoltVision) share this route: nemotron_h LLM + C-RADIO tower, no sound tower.
    return (cfg.get("model_type") in ("nemotron_h", "nemotron_h_v2") and vc.get("model_type") == "radio"
            and omni.get("sound_config") is None and "img_context_token_id" in omni)


class LanguageModel(nn.Module):
    """Text side for the MLLM scheduler. With a native MTP head (Nemotron-3.5-Super-VL) it also exposes the native-MTP
    contract the generator drives: ``return_hidden`` (the backbone output AFTER norm_f — the h_t the head was trained
    on; before norm_f is wrong, jang Super 01-LOG control), ``mtp_forward`` and ``make_mtp_cache``. With
    ``n_confirmed`` the Mamba-2 layers publish rollback points for the verify forward (models/nemotron_mamba_rollback.py)
    so a rejected draft rolls back instead of replaying the confirmed token."""

    def __init__(self, inner, mtp=None):
        super().__init__()
        self.inner = inner
        self.mtp = mtp

    @property
    def layers(self):
        return self.inner.layers

    def make_cache(self):
        return self.inner.make_cache()

    def make_mtp_cache(self):
        return [KVCache()]                                  # the head's attention sub-layer keeps its own KV

    def mtp_forward(self, hidden, next_tokens, mtp_cache, return_hidden=False):
        """One head step: pairs (h_i, t_{i+1}) -> logits for t_{i+2}. Several pairs in one call (the aligned head cache
        commits every confirmed pair) need a CAUSAL mask over the head's attention; jang_tools' __call__ passes none, so
        the sub-layers are driven here."""
        from mlx_lm.models.base import create_attention_mask

        embed = self.inner.backbone.embeddings(next_tokens)
        if hidden.shape[1] != embed.shape[1]:
            hidden = hidden[:, -embed.shape[1]:, :]
        cache = mtp_cache[0] if mtp_cache else None
        entry, moe = self.mtp.layers
        x = entry.fuse(embed, hidden)
        x = entry(x, mask=create_attention_mask(x, cache), cache=cache)
        h = moe.final_layernorm(moe(x))
        logits = self.inner.lm_head(h)
        return (logits, h) if return_hidden else logits

    def __call__(self, inputs, cache=None, inputs_embeds=None, return_hidden=False, n_confirmed=0, **kwargs):
        if n_confirmed and cache is not None and self.mtp is not None:
            from . import nemotron_mamba_rollback as rb
            marked = rb.mark_confirmed(cache, n_confirmed)
            try:
                return self._forward(inputs, cache, inputs_embeds, return_hidden)
            finally:
                rb.unmark(marked)
        return self._forward(inputs, cache, inputs_embeds, return_hidden)

    def _forward(self, inputs, cache, inputs_embeds, return_hidden):
        bb = self.inner.backbone
        if inputs_embeds is None:
            if not return_hidden:
                return LanguageModelOutput(logits=self.inner(inputs, cache=cache))
            h = bb(inputs, cache=cache)
        else:
            h = _backbone_from_embeddings(bb, inputs_embeds, cache)
        logits = self.inner.lm_head(h)
        if return_hidden:
            return logits, h
        return LanguageModelOutput(logits=logits)


def _backbone_from_embeddings(bb, h, cache):
    """mlx_lm NemotronHModel.__call__ with the embedding lookup replaced by precomputed rows."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    if cache is None:
        cache = [None] * len(bb.layers)
    attn_mask = create_attention_mask(h, cache[bb.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[bb.ssm_idx])
    ci = 0
    for layer in bb.layers:
        c = None
        if layer.block_type in ("M", "*"):
            c = cache[ci]; ci += 1
        h = layer(h, mask=attn_mask if layer.block_type == "*" else ssm_mask, cache=c)
    return bb.norm_f(h)


class Model(nn.Module):
    def __init__(self, text, bundle, config, mtp=None):
        super().__init__()
        self.language_model = LanguageModel(text, mtp)
        self._bundle = Path(bundle)
        self._vision = None
        omni = json.loads((self._bundle / "config_omni.json").read_text())
        self.image_token_id = int(omni["img_context_token_id"])
        # The placeholder id lives in config_omni.json only; the generator finds media runs (per-image cache keys,
        # media-aware chunk edges) from ``image_token_id`` on the wrapper config, so expose it there.
        self.config = SimpleNamespace(**{**config, "image_token_id": self.image_token_id})
        self.model_type = "nemotron_h"
        # Media-conditioned prefix reuse (see mllm_batch_generator._mllm_media_prefix_cache_family_enabled).
        self._nemotron_bolt_vl_runtime = True

    def vision(self):
        if self._vision is None:
            from jang_tools.nemotron_bolt.vision import BoltVision
            self._vision = BoltVision(self._bundle)
        return self._vision

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(self, input_ids, pixel_values=None, image_grid_thw=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Nemotron-Bolt media requires one request per prefill")
        emb = self.language_model.inner.backbone.embeddings(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=emb)
        ids = np.array(input_ids[0])
        grid = np.asarray(image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw).reshape(-1, 3)
        flat = pixel_values.reshape(-1)
        rows, off = [], 0
        for _, H, W in grid.tolist():
            n = 3 * int(H) * int(W)
            rows.append(self.vision().embed_pixels(np.asarray(flat[off:off + n].astype(mx.float32)).reshape(1, 3, int(H), int(W))))
            off += n
        if off != flat.size:
            raise ValueError("Nemotron-Bolt pixel buffer does not match the image grid")
        feats = mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
        positions = np.flatnonzero(ids == self.image_token_id)
        if feats.shape[0] != len(positions):
            raise ValueError(f"Nemotron-Bolt: {feats.shape[0]} visual rows vs {len(positions)} <image> placeholders")
        emb[0, mx.array(positions)] = feats.astype(emb.dtype)
        return InputEmbeddingsFeatures(inputs_embeds=emb)

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values=pixel_values, **kwargs)
        return self.language_model(input_ids, cache=cache, inputs_embeds=features.inputs_embeds)


class NemotronBoltProcessor:
    """Tokenizer-compatible processor: the bundle's chat template renders ``<image>`` per image part; the bundle's own
    HF processor expands each to ``<img>`` + N ``<image>`` + ``</img>`` and prepares the dynamic-resolution pixels."""

    model_type = "nemotron_h"

    def __init__(self, tokenizer, bundle):
        self.tokenizer, self.bundle = tokenizer, Path(bundle)
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self._hf = None

    def __getattr__(self, name):
        if name == "process":
            raise AttributeError(name)
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _proc(self):
        if self._hf is None:
            from transformers import AutoProcessor
            self._hf = AutoProcessor.from_pretrained(str(self.bundle), trust_remote_code=True)
        return self._hf

    def __call__(self, text, images=None, videos=None, audio=None, add_special_tokens=False, return_tensors="mlx", **kwargs):
        if videos or audio:
            raise ValueError("Nemotron-Bolt on vMLX: video and audio inputs are not supported yet")
        if isinstance(text, list):
            if len(text) != 1:
                raise ValueError("Nemotron-Bolt media processor accepts one prompt at a time")
            text = text[0]
        images = list(images or [])
        if not images:
            ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return {"input_ids": mx.array([ids], dtype=mx.int32), "attention_mask": mx.ones((1, len(ids)), dtype=mx.int32)}
        from mlx_vlm.utils import load_image
        pil = [load_image(v) if isinstance(v, (str, Path)) else v for v in images]
        pil = [p.convert("RGB") for p in pil]
        # dynamic resolution: images of different sizes have different pixel shapes, so never ask the processor to stack
        # them into one tensor (return_tensors="np" fails on a 2-image turn with mixed sizes — found by the variating
        # gate: image in history + a new chart image + tools in the same turn)
        enc = self._proc()(text=[text], images=pil, return_tensors=None)
        ids = [int(t) for t in np.asarray(enc["input_ids"][0]).reshape(-1)]
        pvs = enc["pixel_values"]
        pvs = list(pvs) if isinstance(pvs, (list, tuple)) else [pvs[i:i + 1] for i in range(pvs.shape[0])]
        flat, grid = [], []
        for pv in pvs:
            pv = np.asarray(pv, dtype=np.float32)
            pv = pv if pv.ndim == 4 else pv[None]
            if pv.shape[0] != 1:
                raise ValueError("Nemotron-Bolt: one pixel tensor per image expected")
            grid.append([1, pv.shape[2], pv.shape[3]]); flat.append(pv.reshape(-1))
        return {"input_ids": mx.array([ids], dtype=mx.int32), "attention_mask": mx.ones((1, len(ids)), dtype=mx.int32),
                "pixel_values": mx.array(np.concatenate(flat)), "image_grid_thw": mx.array(np.asarray(grid, np.int32))}


def _quantization_from_tensors(weights: dict, path: str, in_features: int):
    """(bits, group_size) of a stored affine Linear / SwitchLinear, from its packed width against the module's true
    input width (the packed width alone is ambiguous: 256 uint32 words fit 8b/g32, 4b/g64 and 2b/g128), or None when
    the module is stored unquantized."""
    w, s = weights.get(path + ".weight"), weights.get(path + ".scales")
    if w is None or s is None:
        return None
    packed_bits = int(w.shape[-1]) * 32
    groups = int(s.shape[-1])
    if packed_bits % in_features or in_features % groups:
        raise ValueError(f"nemotron MTP: {path} packed {tuple(w.shape)} / scales {tuple(s.shape)} do not fit {in_features} inputs")
    bits, group = packed_bits // in_features, in_features // groups
    if bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128):
        raise ValueError(f"nemotron MTP: {path} resolves to unsupported {bits}-bit / group {group}")
    return bits, group


def load_mtp_head(text, bundle):
    """Build NVIDIA's Nemotron-H MTP head (jang_tools.nemotron_mtp structure: eh_proj fusion + attention sub-layer +
    MoE sub-layer + final_layernorm, sharing embeddings and lm_head) and load only the ``mtp.*`` tensors, from the
    shards that hold them. Each module is quantized exactly as stored (bits / group size inferred from its tensors)."""
    from jang_tools.nemotron_mtp import NemotronHMTP

    bundle = Path(bundle)
    index = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({f for k, f in index.items() if k.startswith("mtp.")})
    weights = {}
    for shard in shards:
        for k, v in mx.load(str(bundle / shard)).items():
            if k.startswith("mtp."):
                weights[k[len("mtp."):]] = v
    if not weights:
        return None
    head = NemotronHMTP(text.args)
    specs = {}
    for name, module in head.named_modules():
        if isinstance(module, nn.Linear) or type(module).__name__ in ("SwitchLinear",):
            q = _quantization_from_tensors(weights, name, int(module.weight.shape[-1]))
            if q is not None:
                specs[name] = q

    def predicate(path, module):
        q = specs.get(path)
        return {"bits": q[0], "group_size": q[1]} if q else False

    nn.quantize(head, class_predicate=predicate)
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    logger.info("Nemotron-H native MTP head loaded: %d tensors from %d shard(s), %d quantized modules (%s)",
                len(weights), len(shards), len(specs),
                ", ".join(sorted({f"{b}b/g{g}" for b, g in specs.values()})))
    return head


def load_nemotron_bolt_vl(bundle):
    from ..native_mtp import maybe_apply_native_mtp
    from ..utils.jang_loader import load_jang_model
    text, tokenizer = load_jang_model(str(bundle))
    config = json.loads((Path(bundle) / "config.json").read_text())
    mtp = None
    if int(config.get("num_nextn_predict_layers") or 0) > 0:
        status = maybe_apply_native_mtp(str(bundle), allow_runtime=True)
        if status.get("runtime_active"):
            mtp = load_mtp_head(text, bundle)
            if mtp is not None:
                from . import nemotron_mamba_rollback
                nemotron_mamba_rollback.apply()
    model = Model(text, bundle, config, mtp=mtp)
    return model, NemotronBoltProcessor(tokenizer, bundle), config

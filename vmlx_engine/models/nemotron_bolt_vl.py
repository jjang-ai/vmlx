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
import os
import weakref

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


class _TapRecorder:
    """Rolling record of the drafter's target taps (outputs of target blocks layer_ids, before the final norm) for the
    last ``window`` absolute positions. Every target forward writes its rows at [start, start + L); a write truncates
    any older rows from ``start`` on, so after a rejected verify the rewritten positions always hold the committed
    prefix's taps (latest write wins)."""

    def __init__(self, n_taps, window):
        self.n, self.window = int(n_taps), int(window)
        self.slots = [None] * self.n
        self._owner = None
        self._contexts = weakref.WeakKeyDictionary()
        self._unowned_chunks = []

    def select(self, owner):
        """Keep auxiliary forwards and restored conversations in separate histories.

        The cache owns its history's lifetime. Neither the active selection nor
        the history index keeps a completed request's KV/SSM cache alive.
        """
        previous = self._owner() if self._owner is not None else None
        if previous is not owner:
            self.slots = [None] * self.n
        self._owner = weakref.ref(owner) if owner is not None else None

    @property
    def chunks(self):
        if self._owner is None:
            return self._unowned_chunks
        owner = self._owner()
        return self._contexts.get(owner, []) if owner is not None else []

    @chunks.setter
    def chunks(self, value):
        if self._owner is None:
            self._unowned_chunks = value
        elif (owner := self._owner()) is not None:
            self._contexts[owner] = value

    def put(self, slot, y):
        self.slots[slot] = y

    def flush(self, start):
        if any(t is None for t in self.slots) or start is None:
            self.slots = [None] * self.n
            return None
        taps = mx.concatenate(self.slots, axis=-1)              # (1, L, n*D)
        self.slots = [None] * self.n
        L = int(taps.shape[1])
        self.chunks = [(s, a[: max(0, start - s)]) for s, a in self.chunks if s < start]
        self.chunks = [(s, a) for s, a in self.chunks if a.shape[0] > 0]
        keep = taps[0, max(0, L - self.window):]
        self.chunks.append((start + L - keep.shape[0], keep))
        end = start + L
        pruned = []
        for s, a in self.chunks:                                # keep only the last `window` positions
            lo = max(s, end - self.window)
            if lo < s + a.shape[0]:
                pruned.append((lo, a[lo - s:]))
        self.chunks = pruned
        return taps

    def rows(self, lo, hi):
        """(start, rows) for positions [max(lo, earliest), hi) that are recorded contiguously up to hi, else (None, None)."""
        parts, pos = [], None
        for s, a in self.chunks:
            e = s + a.shape[0]
            if e <= lo or s >= hi:
                continue
            a0 = max(lo, s)
            if pos is not None and a0 != pos:
                parts, pos = [], None                           # gap: restart after it
            if pos is None:
                start = a0
            parts.append(a[a0 - s: min(hi, e) - s])
            pos = min(hi, e)
        if not parts or pos != hi:
            return None, None
        return start, mx.concatenate(parts, axis=0)


_TAP_HOOK_INSTALLED = False


def _install_block_tap_hook():
    """NemotronHBlock records its output into an attached sink. A class-level hook keeps the block modules in the
    parameter tree (swapping list entries for proxies would hide their weights from parameter walks)."""
    global _TAP_HOOK_INSTALLED
    if _TAP_HOOK_INSTALLED:
        return
    from mlx_lm.models import nemotron_h

    Block = nemotron_h.NemotronHBlock
    original = Block.__call__

    def call(self, x, *args, **kwargs):
        y = original(self, x, *args, **kwargs)
        sink = getattr(self, "_vmlx_tap_sink", None)
        if sink is not None:
            sink.put(self._vmlx_tap_slot, y)
        return y

    Block.__call__ = call
    _TAP_HOOK_INSTALLED = True


class _LiLiCorrHeadCache:
    """mtp_cache element of the LiLiCorr adapter: the drafted path of the current block. The drafter's context lives in
    the drafter and is synced from the tap recorder, so trims and head-cache recreation never lose committed history."""

    def __init__(self):
        self.path, self.level, self.offset = [], 0, 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        return int(n)                                           # chain pairs never enter the drafter context


class LiLiCorrLanguageModel(LanguageModel):
    """Nemotron-3.5-Bolt text side with NVIDIA's LiLiCorr block drafter behind the native-MTP head contract:
    * every target forward records the drafter's taps (``_TapRecorder``); ``return_hidden`` returns them;
    * level-0 ``mtp_forward`` (taps as hidden): sync the drafter context to every committed position (the target
      attention cache offset), draft the block [anchor, mask x7] at that position, LiLiCorr greedy path; returns the
      path's first token as a point-mass proposal (rejection sampling stays exact: q(x) = 1);
    * deeper levels (the marker hidden) return the next path token.
    Mamba-2 rollback points + row-exact verify come from models/nemotron_mamba_rollback.py (``n_confirmed``)."""

    # every mtp_forward returns point-mass logits (the lattice path token, -1e9 elsewhere): the shared MTP draft step
    # takes the argmax directly instead of running the request sampler over the vocabulary (mllm_batch_generator)
    _vmlx_point_mass_draft = True

    # Loaded-runtime identity for health; this is not a native mtp.* artifact.
    _vmlx_mtp_runtime_adapter = "nemotron_lilicorr"

    def __init__(self, inner, drafter):
        super().__init__(inner, mtp=drafter)
        self.drafter = drafter
        self._tap_dim = len(drafter.layer_ids) * int(inner.backbone.layers[0].norm.weight.shape[0])
        self._rec = _TapRecorder(len(drafter.layer_ids), drafter.window)
        self._last_cache_ref = None
        self._ctx_owner = None
        _install_block_tap_hook()
        for slot, idx in enumerate(drafter.layer_ids):
            layer = inner.backbone.layers[idx]
            object.__setattr__(layer, "_vmlx_tap_sink", self._rec)
            object.__setattr__(layer, "_vmlx_tap_slot", slot)

    @staticmethod
    def _attn_offset(cache):
        for c in cache or []:
            if isinstance(c, KVCache):
                return int(c.offset)
        return None

    def make_mtp_cache(self):
        return [_LiLiCorrHeadCache()]

    def _forward(self, inputs, cache, inputs_embeds, return_hidden):
        anchor = next((c for c in cache or [] if isinstance(c, KVCache)), None)
        start = int(anchor.offset) if anchor is not None else None
        self._rec.select(anchor)
        if return_hidden and cache is not None:
            # only the MTP seed / verify forwards ask for hidden, and they run on the request's own cache; auxiliary
            # forwards (clean-store re-prefills on a temporary cache) must not move the anchor position
            self._last_cache_ref = weakref.ref(anchor) if anchor is not None else None
        try:
            out = super()._forward(inputs, cache, inputs_embeds, return_hidden)
            taps = self._rec.flush(start)
        finally:
            self._rec.slots = [None] * self._rec.n
        if return_hidden:
            logits, _ = out
            return logits, taps
        return out

    def _sync_context(self, p):
        d = self.drafter
        owner = self._last_cache_ref() if self._last_cache_ref is not None else None
        if owner is None:
            return False
        self._rec.select(owner)
        previous = self._ctx_owner() if self._ctx_owner is not None else None
        if previous is not owner or d.ctx_end > p:
            d.reset()
            self._ctx_owner = weakref.ref(owner)
        lo = d.ctx_end if d.ctx_end > 0 else max(0, p - d.window)
        if lo >= p:
            return True
        start, rows = self._rec.rows(lo, p)
        if rows is None:
            return False
        if d.ctx_end != start:                                  # history before `start` is not recorded (cached prefix)
            d.reset()
            d.ctx_end = start
        d.add_context(rows, start)
        return True

    def mtp_forward(self, hidden, next_tokens, mtp_cache, return_hidden=False):
        hc = mtp_cache[0]
        vocab = int(self.inner.lm_head.weight.shape[0]) if hasattr(self.inner.lm_head, "weight") else None
        if hidden.shape[-1] == self._tap_dim:
            anchor_cache = self._last_cache_ref() if self._last_cache_ref is not None else None
            p = int(anchor_cache.offset) if anchor_cache is not None else None
            anchor = int(next_tokens.reshape(-1)[-1].item())
            hc.path, hc.level = [], 0
            if p is not None and self._sync_context(p):
                d = self.drafter
                emb = self.inner.backbone.embeddings
                h = d.draft_hidden(emb(mx.array([[anchor] + [d.mask_id] * (d.block - 1)])), p)
                path, _top1 = d.propose(h, self.inner.lm_head, emb)
                hc.path = path or _top1
            tok = hc.path[0] if hc.path else anchor
            hc.level = 1
        else:
            tok = hc.path[hc.level] if hc.level < len(hc.path) else (hc.path[-1] if hc.path else 0)
            hc.level += 1
        if vocab is None:
            vocab = int(self.inner.args.vocab_size)
        logits = mx.full((1, 1, vocab), -1e9, dtype=mx.float32)
        logits[0, 0, tok] = 0.0
        marker = mx.zeros((1, 1, 1), dtype=mx.float32)
        return (logits, marker) if return_hidden else logits


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
    def __init__(self, text, bundle, config, mtp=None, language_model=None):
        super().__init__()
        self.language_model = language_model if language_model is not None else LanguageModel(text, mtp)
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
    language_model = None
    from ..native_mtp import set_native_mtp_drafter_max_depth

    set_native_mtp_drafter_max_depth(None)
    lilicorr_dir = Path(bundle) / "lilicorr"
    if mtp is None and lilicorr_dir.is_dir() and os.environ.get("VMLX_NEMOTRON_LILICORR", "0").strip() == "1":
        # Opt-in until a live A/B qualifies it (offline greedy tau 2.8-4.9 on this target, ledger B12).
        from . import nemotron_mamba_rollback
        from .nemotron_lilicorr import LiLiCorrDrafter

        drafter = LiLiCorrDrafter(lilicorr_dir)
        nemotron_mamba_rollback.apply()
        language_model = LiLiCorrLanguageModel(text, drafter)
        # One drafter pass proposes block - 1 path tokens, verified byte-exact at D7 (ledger B13): Adaptive's ladder
        # may climb there instead of stopping at the recursive-head default D3.
        set_native_mtp_drafter_max_depth(int(drafter.block) - 1)
        logger.info("Nemotron-3.5-Bolt LiLiCorr drafter attached (block %d, taps %s, window %d)",
                    drafter.block, drafter.layer_ids, drafter.window)
    model = Model(text, bundle, config, mtp=mtp, language_model=language_model)
    return model, NemotronBoltProcessor(tokenizer, bundle), config

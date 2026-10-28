"""DSpark drafter (Qwen3DSparkModel) in MLX.

DFlash block drafter + DSpark Markov / confidence heads, as released with
openbmb MiniCPM-V-4.7 (``model_type: qwen3``, ``dflash_config.projector_type
dspark``). Semantics follow vLLM ``qwen3_dflash.py`` / ``qwen3_dspark.py``:

- context: outputs of the target decoder layers ``target_layer_ids`` (residual
  stream, before the final norm), concatenated -> ``fc`` -> ``hidden_norm``; each
  draft layer projects it with its own ``k_proj`` / ``v_proj`` (no input norm),
  ``k_norm``, RoPE at the context positions; cached across rounds;
- block: ``[anchor, mask x (block_size - 1)]`` embedded by the TARGET's
  ``embed_tokens`` (the mask is the vocabulary row ``mask_token_id``; the
  checkpoint has no ``mask_embedding`` / ``embed_tokens`` / ``lm_head``);
- 5 Qwen3 layers (standard ``w * x`` RMSNorm, q/k norm, full rotary over
  ``rope_dim``), attention non-causal: queries = block, keys = context + block;
- logits: ``norm`` -> the TARGET's ``lm_head``; block position i predicts the
  token after it (draft token i + 1);
- Markov head: ``logits_i += markov_w2 @ markov_w1[prev_i]`` with prev_0 = the
  anchor and prev_i = the token drafted at i - 1 (sequential);
- confidence head: ``sigmoid(proj([h_i ; markov_w1[prev_i]]))`` in float32.

MiniMax-M3.1 flavor (``dspark/`` in the MiniMax-M3.1-Flash release,
``DSparkMiniMaxDraftModel``): the same block drafter with MiniMax layers, i.e.
Gemma-style ``(1 + w)`` norms (also on the per-head q / k norms, ``hidden_norm``
and the final norm), partial RoPE over ``rotary_dim`` of ``head_dim``, a dense
SwiGLU-OAI FFN, and sliding-window attention: a block row at position p reads
context keys at positions > p - ``sliding_window`` plus the whole block. The
checkpoint's linears are MXFP8 (kept as stored); ``fc`` and both heads are BF16.
A pre-quantized copy (``write_quantized_minimax_drafter``) declares its formats
in ``config.json["quantization"]``, mlx-lm's per-module form keyed by the live
module paths (``layers.N.self_attn.q_proj``, ``fc``, ``markov_head.markov_w2``):
affine and MXFP8 modules may be mixed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class _Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, heads, kv_heads, head_dim = (cfg["hidden_size"], cfg["num_attention_heads"],
                                             cfg["num_key_value_heads"], cfg["head_dim"])
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = _norm(cfg, head_dim)
        self.k_norm = _norm(cfg, head_dim)


class _MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self._oai = (cfg["swiglu_alpha"], cfg["swiglu_limit"]) if cfg.get("flavor") == "minimax" else None

    def __call__(self, y: mx.array) -> mx.array:
        if self._oai is None:
            return self.down_proj(nn.silu(self.gate_proj(y)) * self.up_proj(y))
        from ..models.minimax_m3.minimax_m3 import swiglu

        return self.down_proj(swiglu(self.up_proj(y), self.gate_proj(y), *self._oai))


class _Layer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.self_attn = _Attention(cfg)
        self.mlp = _MLP(cfg)
        self.input_layernorm = _norm(cfg, hidden)
        self.post_attention_layernorm = _norm(cfg, hidden)


class _Markov(nn.Module):
    def __init__(self, vocab: int, draft_vocab: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, draft_vocab, bias=False)


class _Confidence(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.proj = nn.Linear(dims, 1, bias=True)


def is_minimax_config(raw: dict) -> bool:
    return "DSparkMiniMaxDraftModel" in (raw.get("architectures") or []) or "dspark_block_size" in (
        raw.get("text_config") or raw)


def minimax_config(raw: dict) -> dict:
    """The MiniMax-M3.1 draft config in this module's flat schema (``flavor: minimax``)."""
    t = raw.get("text_config") or raw
    dsc = t.get("dspark_config") or {}
    problems = []
    if str(t.get("dspark_markov_head_type", dsc.get("markov_head_type", "vanilla"))) != "vanilla":
        problems.append("markov head is not vanilla")
    if not t.get("dspark_use_dense_ffn", dsc.get("use_dense_ffn", True)):
        problems.append("MoE draft FFN")
    if t.get("dspark_hybrid_context_fusion", dsc.get("hybrid_context_fusion", False)):
        problems.append("hybrid context fusion")
    if any(lt != "sliding_attention" for lt in t.get("layer_types") or []):
        problems.append("non-sliding draft layers")
    if not (raw.get("enable_confidence_head", t.get("enable_confidence_head"))
            and raw.get("confidence_head_with_markov", t.get("confidence_head_with_markov", True))):
        problems.append("confidence head without the Markov latent")
    if problems:
        raise ValueError("MiniMax DSpark drafter not supported: " + ", ".join(problems))
    head_dim = int(t["head_dim"])
    return {
        "flavor": "minimax",
        "hidden_size": int(t["hidden_size"]),
        "num_hidden_layers": int(t["num_hidden_layers"]),
        "num_attention_heads": int(t["num_attention_heads"]),
        "num_key_value_heads": int(t["num_key_value_heads"]),
        "head_dim": head_dim,
        "rotary_dim": int(t.get("rotary_dim") or head_dim * float(t.get("partial_rotary_factor", 1.0))),
        "rms_norm_eps": float(t.get("rms_norm_eps", 1e-6)),
        "vocab_size": int(t["vocab_size"]),
        "intermediate_size": int(t.get("dspark_ffn_hidden_size") or dsc.get("ffn_hidden_size")
                                 or t["dense_intermediate_size"]),
        "swiglu_alpha": float(t.get("swiglu_alpha", 1.702)),
        "swiglu_limit": float(t.get("swiglu_limit", 7.0)),
        "sliding_window": int(t["sliding_window"]),
        "block_size": int(t["dspark_block_size"]),
        "rope_parameters": {"rope_theta": float(t["rope_theta"])},
        "enable_confidence_head": True,
        "dflash_config": {
            "mask_token_id": int(t.get("dspark_noise_token_id", dsc.get("mask_token_id"))),
            "target_layer_ids": [int(i) for i in t["dspark_target_layer_ids"]],
            "markov_rank": int(t.get("dspark_markov_rank", dsc.get("markov_rank"))),
        },
    }


def _norm(cfg: dict, dims: int) -> nn.Module:
    if cfg.get("flavor") == "minimax":
        from ..models.minimax_m3.minimax_m3 import GemmaRMSNorm

        return GemmaRMSNorm(dims, cfg["rms_norm_eps"])
    return nn.RMSNorm(dims, eps=cfg["rms_norm_eps"])


def _check_config(cfg: dict) -> None:
    if cfg.get("flavor") == "minimax":
        return  # checked in minimax_config
    dc = cfg.get("dflash_config") or {}
    problems = []
    if dc.get("projector_type") != "dspark":
        problems.append(f"projector_type={dc.get('projector_type')!r}")
    if dc.get("attention_mode", "gqa") != "gqa":
        problems.append(f"attention_mode={dc.get('attention_mode')!r}")
    if dc.get("markov_head_type", "vanilla") != "vanilla":
        problems.append(f"markov_head_type={dc.get('markov_head_type')!r}")
    if not cfg.get("enable_confidence_head", dc.get("enable_confidence_head")):
        problems.append("no confidence head")
    if any(t != "full_attention" for t in cfg.get("layer_types") or []):
        problems.append("sliding-window draft layers")
    if dc.get("causal") or cfg.get("is_causal"):
        problems.append("causal draft attention")
    rope = cfg.get("rope_parameters") or {}
    if int(rope.get("rope_dim", cfg["head_dim"])) != int(cfg["head_dim"]):
        problems.append("partial rotary")
    if int(cfg.get("draft_vocab_size") or cfg["vocab_size"]) != int(cfg["vocab_size"]):
        problems.append("reduced draft vocabulary (needs d2t)")
    if problems:
        raise ValueError("DSpark drafter not supported: " + ", ".join(problems))


class DSparkDrafter(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        _check_config(cfg)
        self.cfg = cfg
        dc = cfg["dflash_config"]
        hidden = cfg["hidden_size"]
        self.block = int(cfg["block_size"])
        self.mask_token_id = int(dc["mask_token_id"])
        self.target_layer_ids = [int(i) for i in dc["target_layer_ids"]]
        self.head_dim = int(cfg["head_dim"])
        self.rope_dims = int(cfg.get("rotary_dim") or self.head_dim)
        self.rope_theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", 1e7))
        # Model-defined sliding window (MiniMax); 0 = the whole context.
        self.sliding_window = int(cfg.get("sliding_window") or 0)
        self.layers = [_Layer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.fc = nn.Linear(len(self.target_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = _norm(cfg, hidden)
        self.norm = _norm(cfg, hidden)
        rank = int(cfg.get("markov_rank") or dc["markov_rank"])
        vocab = int(cfg["vocab_size"])
        self.markov_head = _Markov(vocab, int(cfg.get("draft_vocab_size") or vocab), rank)
        self.confidence_head = _Confidence(hidden + rank)
        self.reset()

    # ------------------------------------------------------------ context
    def reset(self) -> None:
        from mlx_lm.models.cache import KVCache

        # Context K/V per drafter layer, grown in steps (KVCache) instead of concatenated per round: the old
        # concatenation copied the whole context twice per layer per round, a cost that grew with the conversation.
        # Also the form the session store / SSD tier persist and restore (the mlx-lm cache state protocol).
        self._ctx = [KVCache() for _ in self.layers]
        self.ctx_len = 0

    def context_caches(self) -> list:
        """The context K/V caches (one per drafter layer) covering ``ctx_len`` positions."""
        return self._ctx

    def set_context(self, caches: list) -> None:
        """Adopt restored context caches (session store / SSD); they must all cover the same positions."""
        caches = list(caches)
        if len(caches) != len(self.layers):
            raise ValueError("DSpark drafter context: one cache per drafter layer expected")
        offsets = {int(getattr(c, "offset", 0) or 0) for c in caches}
        if len(offsets) != 1:
            raise ValueError(f"DSpark drafter context caches disagree on length: {sorted(offsets)}")
        self._ctx = caches
        self.ctx_len = offsets.pop()

    def context_arrays(self) -> list:
        return [a for c in self._ctx for a in (c.keys, c.values) if a is not None]

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(x, self.rope_dims, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)

    def add_context(self, features: mx.array) -> None:
        """Append target tap features (1, C, n_taps * H) at positions [ctx_len, ctx_len + C)."""
        count = int(features.shape[1])
        if count == 0:
            return
        ctx = self.hidden_norm(self.fc(features))
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            k = attn.k_norm(attn.k_proj(ctx).reshape(1, count, attn.kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
            k = self._rope(k, self.ctx_len)
            v = attn.v_proj(ctx).reshape(1, count, attn.kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            self._ctx[li].update_and_fetch(k, v)
        self.ctx_len += count

    # -------------------------------------------------------------- block
    def __call__(self, block_embeds: mx.array) -> mx.array:
        """block_embeds: (1, block, H) = [anchor, mask...]; positions start at ctx_len."""
        h = block_embeds
        length = int(h.shape[1])
        start = self.ctx_len
        window = _draft_window()
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(1, length, attn.heads, attn.head_dim)).transpose(0, 2, 1, 3)
            k = attn.k_norm(attn.k_proj(x).reshape(1, length, attn.kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(1, length, attn.kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            q = self._rope(q, start)
            k = self._rope(k, start)
            # Context + this block: the block rows are written after the context in the cache buffer and trimmed
            # off again below (the next add_context overwrites them), so no per-round copy of the context.
            k, v = self._ctx[li].update_and_fetch(k, v)
            self._ctx[li].trim(length)
            mask = None
            if self.sliding_window:
                k, v, mask = _sliding(k, v, start, length, self.sliding_window, q.dtype)
            elif window and int(k.shape[2]) > window + length:
                # Opt-in recent-context window (TensorFold's DSV4 drafter attends the last 128 rows): the draft
                # block's attention reads O(window) instead of the whole conversation. Draft-side only.
                k, v = k[:, :, -(window + length):], v[:, :, -(window + length):]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.head_dim**-0.5, mask=mask)
            h = h + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(1, length, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.norm(h)

    def draft(self, block_hidden: mx.array, anchor: mx.array, lm_head, count: int, sampler=None,
              candidates: Optional[int] = None):
        """Sequential Markov chain over block positions 0..count-1.

        Returns (tokens (count,), confidences (count,) float32). With
        ``sampler`` (log-prob -> token) the chain samples instead of argmax.

        ``candidates`` K (default ``VMLX_DSPARK_MARKOV_CANDIDATES``, 32; 0 = full vocabulary): the Markov rescoring
        runs on the top-K base-logit candidates of each position only (Naive N0.5 prototype ``draft_cand``, 2026-10-09).
        The full-vocab bias is one (rank x vocab) matvec per drafted position, sequentially -- on MiniCPM-V-4.7 a
        256 x 248,144 bf16 matrix, 127 MB per position, ~0.9 GB per 7-token draft -- and becomes two K-row gathers.
        It differs from the full chain only when the bias would promote a token outside the base top-K. Drafts are
        always verified by the target, so this changes acceptance, never the output.
        """
        k = _markov_candidates() if candidates is None else int(candidates)
        if 0 < k < int(self.markov_head.markov_w2.weight.shape[0]):
            return self._draft_candidates(block_hidden, anchor, lm_head, count, sampler, k)
        logits = lm_head(block_hidden[:, :count, :])[0].astype(mx.float32)
        w1 = self.markov_head.markov_w1
        w2 = self.markov_head.markov_w2
        proj = self.confidence_head.proj
        conf_w = _float_weight(proj).astype(mx.float32)
        conf_b = proj.bias.astype(mx.float32)
        prev = anchor.reshape(1)
        tokens, confs = [], []
        for i in range(count):
            lat = w1(prev)  # (1, r)
            row = logits[i] + w2(lat)[0].astype(mx.float32)
            feat = mx.concatenate([block_hidden[0, i].astype(mx.float32), lat[0].astype(mx.float32)])
            confs.append(mx.sigmoid(feat @ conf_w[0] + conf_b[0]))
            if sampler is None:
                tok = mx.argmax(row).reshape(1)
            else:
                lp = row - mx.logsumexp(row)
                tok = sampler(lp[None]).reshape(1)
            tokens.append(tok)
            prev = tok
        return mx.concatenate(tokens), mx.stack(confs)


    def _draft_candidates(self, block_hidden, anchor, lm_head, count: int, sampler, k: int):
        logits = lm_head(block_hidden[:, :count, :])[0].astype(mx.float32)          # (count, V)
        cand = mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]                   # (count, K)
        base = mx.take_along_axis(logits, cand, axis=-1)                              # (count, K)
        w1 = self.markov_head.markov_w1                                               # (V, r)
        w2 = self.markov_head.markov_w2                                               # (draft V, r)
        proj = self.confidence_head.proj
        conf_w = _float_weight(proj).astype(mx.float32)
        conf_b = proj.bias.astype(mx.float32)
        prev = anchor.reshape(())
        tokens, confs = [], []
        for i in range(count):
            lat = _rows(w1, prev).astype(mx.float32)                                  # (r,)
            row = base[i] + _rows(w2, cand[i]).astype(mx.float32) @ lat               # (K,)
            feat = mx.concatenate([block_hidden[0, i].astype(mx.float32), lat])
            confs.append(mx.sigmoid(feat @ conf_w[0] + conf_b[0]))
            if sampler is None:
                pick = mx.argmax(row)
            else:
                pick = sampler((row - mx.logsumexp(row))[None]).reshape(())
            tok = cand[i][pick]
            tokens.append(tok.reshape(1))
            prev = tok
        return mx.concatenate(tokens), mx.stack(confs)


def _sliding(k: mx.array, v: mx.array, start: int, length: int, window: int, dtype):
    """Keys for a block at positions [start, start + length) under a model sliding window: context key positions
    > p - window for the row at position p, and the whole block (block attention is non-causal). ``k`` / ``v`` hold
    the context rows [0, start) followed by the block rows."""
    if start + length - 1 - window < 0:  # even the last block row reads every context row
        return k, v, None
    first = max(0, start + 1 - window)  # the earliest context row any block row can read (the anchor row's)
    k, v = k[:, :, first:], v[:, :, first:]
    kpos = mx.concatenate([mx.arange(first, start), mx.full((length,), start + length, dtype=mx.int32)])
    qpos = mx.arange(start, start + length)[:, None]
    keep = kpos[None, :] > qpos - window  # block keys carry start + length, i.e. always inside the window
    return k, v, mx.where(keep, 0.0, -mx.inf).astype(dtype)[None, None]


def _draft_window() -> int:
    """VMLX_DSPARK_DRAFT_WINDOW=N: the draft block attends the last N context rows (0 / unset = all). Changes
    acceptance only (drafts are verified); positions stay absolute."""
    try:
        return max(0, int(os.environ.get("VMLX_DSPARK_DRAFT_WINDOW", "0")))
    except ValueError:
        return 0


def _markov_candidates() -> int:
    try:
        # Default 32 (sustained multi-turn A/Bs 2026-10-13: drafter 3.5 -> 2.4 ms on the 35B; 1B x0.917 vs x0.906).
        return max(0, int(os.environ.get("VMLX_DSPARK_MARKOV_CANDIDATES", "32")))
    except ValueError:
        return 32


QUANT_SKIP = ("markov_head", "confidence_head")


def load_drafter(path: str | Path, bits: Optional[int] = None) -> DSparkDrafter:
    """Load a DSpark drafter directory.

    ``bits``: 8 (default; ``VMLX_DSPARK_DRAFTER_BITS``) quantizes the Linear
    layers to affine 8-bit g64 at load, 16 keeps BF16. A directory written by
    ``write_quantized_drafter`` (``config.json["quantization"]``) loads as
    stored. The Markov and confidence heads always stay BF16.
    """
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    if is_minimax_config(cfg):
        return _load_minimax_drafter(path, cfg)
    model = DSparkDrafter(cfg)
    weights = mx.load(str(path / "model.safetensors"))
    stored = cfg.get("quantization")
    if stored:
        skip = tuple(stored.get("skip", QUANT_SKIP))
        nn.quantize(
            model,
            group_size=int(stored["group_size"]),
            bits=int(stored["bits"]),
            mode=stored.get("mode", "affine"),
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and not p.startswith(skip) and f"{p}.scales" in weights,
        )
        model.load_weights(list(weights.items()), strict=True)
    else:
        if bits is None:
            bits = int(os.environ.get("VMLX_DSPARK_DRAFTER_BITS", "8"))
        model.load_weights(list(weights.items()), strict=True)
        if bits == 8:
            nn.quantize(
                model, group_size=64, bits=8,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and not p.startswith(QUANT_SKIP)
                and m.weight.shape[-1] % 64 == 0,
            )
        elif bits != 16:
            raise ValueError(f"DSpark drafter bits={bits}: 8 or 16 are supported")
    mx.eval(model.parameters())
    return model


def _minimax_weights(path: Path) -> dict:
    """Every tensor of a MiniMax drafter directory under the drafter's module names (the release's
    ``language_model.model.dspark.layers.N.decoder_layer.*`` / ``final_norm.*`` names are mapped; names that already
    are module paths pass unchanged)."""
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    renamed = {}
    for k, v in weights.items():
        k = k.removeprefix("language_model.model.dspark.").replace(".decoder_layer.", ".")
        renamed["norm." + k[len("final_norm."):] if k.startswith("final_norm.") else k] = v
    return renamed


def _load_minimax_drafter(path: Path, raw_cfg: dict) -> DSparkDrafter:
    """A MiniMax-M3.1 ``dspark/`` directory, either
    - as released (no ``quantization`` block): MXFP8 linears (``weight_scale_inv``) kept as stored; fc, Markov and
      confidence heads BF16; or
    - pre-quantized (``config.json["quantization"]``, mlx-lm's per-module form): every module with an entry keyed by
      its live path gets exactly that entry (affine ``{"group_size", "bits"}`` or ``{"mode": "mxfp8", ...}``); a
      module with ``.scales`` but no entry gets the block's top-level ``group_size`` / ``bits`` / ``mode``; MXFP8
      pairs still stored in the release form (``weight_scale_inv``) are MXFP8. Then a strict load."""
    from ..models.minimax_m3.source_format import MXFP8, convert_mxfp8_linears, mxfp8_pairs_to_mlx

    model = DSparkDrafter(minimax_config(raw_cfg))
    renamed = _minimax_weights(path)
    quant = raw_cfg.get("quantization")
    if not isinstance(quant, dict):
        convert_mxfp8_linears(model, renamed)
    else:
        stored_mx8 = mxfp8_pairs_to_mlx(renamed)

        def predicate(p, m):
            if p in stored_mx8:
                return dict(MXFP8)
            if p in quant:
                return quant[p]
            if not hasattr(m, "to_quantized") or f"{p}.scales" not in renamed:
                return False
            return {"group_size": int(quant["group_size"]), "bits": int(quant["bits"]),
                    "mode": quant.get("mode", "affine")}

        nn.quantize(model, class_predicate=predicate)
    model.load_weights(list(renamed.items()), strict=True)
    mx.eval(model.parameters())
    return model


# Default per-module plan of write_quantized_minimax_drafter (fnmatch patterns over live module paths, first match
# wins): decoder-layer linears affine 4-bit, fc affine 8-bit, Markov and confidence heads as stored (BF16).
MINIMAX_DRAFTER_PLAN = {
    "layers.*": {"group_size": 64, "bits": 4},
    "fc": {"group_size": 64, "bits": 8},
    "markov_head.*": None,
    "confidence_head.*": None,
}


def _entry(module) -> Optional[dict]:
    """The quantization entry a module carries (None = float)."""
    if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return None
    if module.mode == "affine":
        return {"group_size": int(module.group_size), "bits": int(module.bits)}
    return {"mode": module.mode, "group_size": int(module.group_size), "bits": int(module.bits)}


def _float_weight(module) -> mx.array:
    """The weight of a Linear / Embedding, dequantized (BF16) when the module is quantized."""
    return _rows(module, None)


def _rows(module, idx) -> mx.array:
    """Rows ``idx`` (None = all) of a Linear / Embedding weight; a quantized module dequantizes only those rows (BF16).
    The Markov / confidence heads are read this way, so pre-quantized heads (MiniMax drafters) draft correctly."""
    if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        flat = None if idx is None else mx.array(idx).reshape(-1)
        take = (lambda a: a) if flat is None else (lambda a: a[flat])
        biases = module.get("biases")
        out = mx.dequantize(take(module.weight), take(module.scales), None if biases is None else take(biases),
                            group_size=module.group_size, bits=module.bits, mode=module.mode, dtype=mx.bfloat16)
        return out if idx is None else out.reshape(*mx.array(idx).shape, -1)
    return module.weight if idx is None else module.weight[idx]


def _write_aligned(flat: dict, out: Path) -> None:
    """One dtype-aligned safetensors shard (jang_tools rewrite_aligned_safetensors), checked."""
    import tempfile

    from jang_tools.format.aligned_safetensors import (
        rewrite_aligned_safetensors,
        verify_safetensors_alignment,
    )

    mx.eval(flat)
    with tempfile.TemporaryDirectory(dir=out.parent.parent) as tmp:
        raw = Path(tmp) / "model.safetensors"
        mx.save_safetensors(str(raw), flat, metadata={"format": "mlx"})
        rewrite_aligned_safetensors(raw, out)
    tensors, unaligned = verify_safetensors_alignment(out)
    if unaligned or tensors != len(flat):
        raise RuntimeError(f"drafter shard alignment check failed: {unaligned} unaligned of {tensors}")


def write_quantized_minimax_drafter(src: str | Path, dst: str | Path, plan: Optional[dict] = None) -> Path:
    """Write a pre-quantized copy of a MiniMax-M3.1 ``dspark/`` directory (never in place).

    ``plan``: fnmatch pattern over live module paths -> target, first match wins (default MINIMAX_DRAFTER_PLAN):
      ``{"group_size": g, "bits": b}``          affine, round to nearest from the stored values (MXFP8 dequantized
                                                 exactly to BF16 first);
      ``{"mode": "mxfp8", "group_size": 32, "bits": 8}``   MXFP8: the stored bytes when the module is stored MXFP8;
      ``None`` / no match                        as stored (MXFP8 bytes or BF16).
    Writes ``model.safetensors`` (module names, dtype-aligned), ``config.json`` = the source config plus
    ``quantization``: top-level ``group_size`` / ``bits`` and one entry per quantized module keyed by its live path,
    and copies every other non-weight file. Drafter quality only moves acceptance (the target verifies every token).
    """
    import fnmatch
    import shutil

    from mlx.utils import tree_flatten

    src, dst = Path(src), Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty")
    raw_cfg = json.loads((src / "config.json").read_text())
    if not is_minimax_config(raw_cfg):
        raise ValueError(f"{src} is not a MiniMax DSpark drafter")
    plan = MINIMAX_DRAFTER_PLAN if plan is None else plan
    model = _load_minimax_drafter(src, raw_cfg)

    def target(path: str, module) -> Optional[dict]:
        for pattern, spec in plan.items():
            if fnmatch.fnmatchcase(path, pattern):
                return _entry(module) if spec is None else dict(spec)
        return _entry(module)

    for path, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, nn.Embedding, nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        want, have = target(path, module), _entry(module)
        if want is not None and want.get("mode", "affine") == "affine":
            want = {"group_size": int(want["group_size"]), "bits": int(want["bits"])}
        if want == have:
            continue                                         # as stored: same bytes
        w = _float_weight(module)
        if isinstance(module, (nn.Embedding, nn.QuantizedEmbedding)):
            fresh = nn.Embedding(w.shape[0], w.shape[1])
        else:
            fresh = nn.Linear(w.shape[1], w.shape[0], bias="bias" in module)
            if "bias" in module:
                fresh.bias = module.bias
        fresh.weight = w
        new = fresh if want is None else fresh.to_quantized(**want)
        parent, _, leaf = path.rpartition(".")
        setattr(_module_at(model, parent) if parent else model, leaf, new)

    quant = {}
    for path, module in model.named_modules():
        e = _entry(module)
        if e is not None:
            quant[path] = e
    affine = [e for e in quant.values() if "mode" not in e]
    top = max(affine, key=lambda e: sum(e == o for o in affine)) if affine else {"group_size": 64, "bits": 8}
    dst.mkdir(parents=True, exist_ok=True)
    _write_aligned(dict(tree_flatten(model.parameters())), dst / "model.safetensors")
    cfg = dict(raw_cfg)
    cfg["quantization"] = {"group_size": top["group_size"], "bits": top["bits"], **dict(sorted(quant.items()))}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in src.iterdir():
        if (f.is_file() and f.name != "config.json" and not f.name.startswith(".")
                and not f.name.endswith((".safetensors", ".safetensors.index.json"))):
            shutil.copy2(f, dst / f.name)
    return dst


def _module_at(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def write_quantized_drafter(src: str | Path, dst: str | Path, *, bits: int = 8, group_size: int = 64,
                            mode: str = "affine") -> Path:
    """Write a pre-quantized copy of a BF16 drafter (never in place).

    Linear layers -> affine ``bits`` / ``group_size``; Markov + confidence
    heads BF16. The shard is written dtype-aligned (jang_tools
    ``rewrite_aligned_safetensors``) and checked; the format is recorded in
    ``config.json["quantization"]``.
    """
    import shutil

    from mlx.utils import tree_flatten

    src, dst = Path(src), Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty")
    dst.mkdir(parents=True, exist_ok=True)
    model = load_drafter(src, bits=16)
    nn.quantize(
        model, group_size=group_size, bits=bits, mode=mode,
        class_predicate=lambda p, m: isinstance(m, nn.Linear) and not p.startswith(QUANT_SKIP)
        and m.weight.shape[-1] % group_size == 0,
    )
    _write_aligned(dict(tree_flatten(model.parameters())), dst / "model.safetensors")
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = {"mode": mode, "bits": bits, "group_size": group_size, "skip": list(QUANT_SKIP)}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in src.iterdir():
        if f.is_file() and f.name not in ("model.safetensors", "config.json") and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)
    return dst

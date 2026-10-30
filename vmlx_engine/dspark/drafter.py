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
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class _Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)


class _MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.self_attn = _Attention(
            hidden, cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], eps
        )
        self.mlp = _MLP(hidden, cfg["intermediate_size"])
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)


class _Markov(nn.Module):
    def __init__(self, vocab: int, draft_vocab: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, draft_vocab, bias=False)


class _Confidence(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.proj = nn.Linear(dims, 1, bias=True)


def _check_config(cfg: dict) -> None:
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
        self.rope_theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", 1e7))
        self.layers = [_Layer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.fc = nn.Linear(len(self.target_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = nn.RMSNorm(hidden, eps=cfg["rms_norm_eps"])
        self.norm = nn.RMSNorm(hidden, eps=cfg["rms_norm_eps"])
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
        return mx.fast.rope(x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)

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
            if window and int(k.shape[2]) > window + length:
                # Opt-in recent-context window (e.g. the last 128 rows): the draft
                # block's attention reads O(window) instead of the whole conversation. Draft-side only.
                k, v = k[:, :, -(window + length):], v[:, :, -(window + length):]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.head_dim**-0.5, mask=None)
            h = h + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(1, length, -1))
            y = layer.post_attention_layernorm(h)
            h = h + layer.mlp.down_proj(nn.silu(layer.mlp.gate_proj(y)) * layer.mlp.up_proj(y))
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
        conf_w = proj.weight.astype(mx.float32)
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
        w1 = self.markov_head.markov_w1.weight                                        # (V, r)
        w2 = self.markov_head.markov_w2.weight                                        # (draft V, r)
        proj = self.confidence_head.proj
        conf_w = proj.weight.astype(mx.float32)
        conf_b = proj.bias.astype(mx.float32)
        prev = anchor.reshape(())
        tokens, confs = [], []
        for i in range(count):
            lat = w1[prev].astype(mx.float32)                                         # (r,)
            row = base[i] + w2[cand[i]].astype(mx.float32) @ lat                      # (K,)
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


def write_quantized_drafter(src: str | Path, dst: str | Path, *, bits: int = 8, group_size: int = 64,
                            mode: str = "affine") -> Path:
    """Write a pre-quantized copy of a BF16 drafter (never in place).

    Linear layers -> affine ``bits`` / ``group_size``; Markov + confidence
    heads BF16. The shard is written dtype-aligned (jang_tools
    ``rewrite_aligned_safetensors``) and checked; the format is recorded in
    ``config.json["quantization"]``.
    """
    import shutil
    import tempfile

    from jang_tools.format.aligned_safetensors import (
        rewrite_aligned_safetensors,
        verify_safetensors_alignment,
    )
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
    flat = dict(tree_flatten(model.parameters()))
    mx.eval(flat)
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        raw = Path(tmp) / "model.safetensors"
        mx.save_safetensors(str(raw), flat, metadata={"format": "mlx"})
        rewrite_aligned_safetensors(raw, dst / "model.safetensors")
    tensors, unaligned = verify_safetensors_alignment(dst / "model.safetensors")
    if unaligned or tensors != len(flat):
        raise RuntimeError(f"drafter shard alignment check failed: {unaligned} unaligned of {tensors}")
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = {"mode": mode, "bits": bits, "group_size": group_size, "skip": list(QUANT_SKIP)}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in src.iterdir():
        if f.is_file() and f.name not in ("model.safetensors", "config.json") and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)
    return dst

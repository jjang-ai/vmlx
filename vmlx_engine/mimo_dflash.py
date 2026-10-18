"""MiMo-V2.6 native DFlash drafter + speculative loop (MLX).

Semantics follow SGLang's DFlash for MiMo (python/sglang/srt/models/dflash.py + speculative/dflash_worker_v2.py, read
2026-10-14), which is what the checkpoint was trained for (Xiaomi's bundled dflash.py is the plain Qwen3 DFlash and lacks
the sinks / V scale / partial RoPE / SWA the checkpoint carries):
  * taps: the OUTPUTS of target layers dflash_config.target_layer_ids (0, 11, 23, 35, 47; the last one pre-final-norm),
    concatenated per token -> fc -> hidden_norm = the context feature;
  * every draft layer: context K/V = k_norm(k_proj(ctx)) with RoPE at the context position, v_proj(ctx) * v_scale (the
    same projected feature feeds all layers); block K/V from the block's own hidden states; per-head q/k RMSNorm;
    RoPE (neox, theta 1e4) on the first head_dim * partial_rotary_factor dims; V scaled by attention_value_scale;
    per-head attention_sink_bias (extra softmax logit, output discarded); NON-causal attention (is_causal false) inside
    a left window of sliding_window (1024) positions;
  * block = [anchor token, mask x (block_size - 1)] embedded with the TARGET embedding, the mask row replaced by the
    trained mask embedding (dflash/mask_embedding.*); the draft proposals are lm_head(norm(h)) of positions 1..B-1.
Rollback: full-attention KVCache.trim; SWA RotatingKVCache after a multi-row verify is in concat mode (window - 1 + rows,
temporal order), so rejected rows are cut from the end (offset and _idx moved back).
Sampling: sample-then-match (each committed token is the target's own draw given the committed prefix).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ------------------------------------------------------------------------------------------------------------ drafter
class DAttn(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.H, self.KV, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        D = c["hidden_size"]
        self.q_proj = nn.Linear(D, self.H * self.hd, bias=False)
        self.k_proj = nn.Linear(D, self.KV * self.hd, bias=False)
        self.v_proj = nn.Linear(D, self.KV * self.hd, bias=False)
        self.o_proj = nn.Linear(self.H * self.hd, D, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, eps=c["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.hd, eps=c["rms_norm_eps"])
        self.attention_sink_bias = mx.zeros((self.H,))
        self.rope_dims = int(self.hd * float(c.get("partial_rotary_factor", 1.0)))
        self.theta = float(c.get("rope_theta", 10000.0))
        dc = c.get("dflash_config") or {}
        self.v_scale = dc.get("attention_value_scale")
        self.scale = self.hd ** -0.5

    def rope(self, x, pos0):
        return mx.fast.rope(x, self.rope_dims, traditional=False, base=self.theta, scale=1.0, offset=pos0)

    def ctx_kv(self, ctx, pos0):
        """ctx (1, n, D) projected context feature -> rope'd K, scaled V (1, KV, n, hd)."""
        n = ctx.shape[1]
        k = self.k_norm(self.k_proj(ctx).reshape(1, n, self.KV, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(ctx).reshape(1, n, self.KV, self.hd).transpose(0, 2, 1, 3)
        if self.v_scale is not None:
            v = v * self.v_scale
        return self.rope(k, pos0), v

    def __call__(self, h, kc, vc, pos0, mask):
        B, S, _ = h.shape
        q = self.q_norm(self.q_proj(h).reshape(B, S, self.H, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(h).reshape(B, S, self.KV, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, S, self.KV, self.hd).transpose(0, 2, 1, 3)
        if self.v_scale is not None:
            v = v * self.v_scale
        q, k = self.rope(q, pos0), self.rope(k, pos0)
        if kc is not None:
            k = mx.concatenate([kc, k], axis=2); v = mx.concatenate([vc, v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask,
                                                 sinks=self.attention_sink_bias.astype(q.dtype))
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, S, -1))


class DMLP(nn.Module):
    def __init__(self, D, I):
        super().__init__()
        self.gate_proj = nn.Linear(D, I, bias=False); self.up_proj = nn.Linear(D, I, bias=False); self.down_proj = nn.Linear(I, D, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.self_attn = DAttn(c)
        self.post_attention_layernorm = nn.RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.mlp = DMLP(c["hidden_size"], c["intermediate_size"])


class MiMoDFlashDrafter(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        dc = c.get("dflash_config") or {}
        self.cfg = c
        self.layer_ids = list(dc["target_layer_ids"])
        self.mask_id = int(dc["mask_token_id"])
        self.block = int(dc.get("block_size", c.get("block_size", 8)))
        self.window = int(c.get("sliding_window") or 1 << 30)
        self.fc = nn.Linear(len(self.layer_ids) * c["hidden_size"], c["hidden_size"], bias=False)
        self.hidden_norm = nn.RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.layers = [DLayer(c) for _ in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.reset()

    def reset(self):
        self.kv = [(None, None) for _ in self.layers]     # context K/V per layer (1, KV, n, hd), last `window` positions
        self.ctx_end = 0                                    # positions [ctx_end - n, ctx_end) are in the context cache

    def add_context(self, taps: mx.array, pos0: int):
        """taps (n, n_layers*D) of target positions pos0..pos0+n-1 (consecutive, appended after the current context)."""
        assert pos0 == self.ctx_end, (pos0, self.ctx_end)
        if taps.shape[0] > self.window:
            pos0 += taps.shape[0] - self.window; taps = taps[-self.window:]
            self.kv = [(None, None) for _ in self.layers]
        ctx = self.hidden_norm(self.fc(taps[None].astype(mx.bfloat16)))
        new = []
        for L, (kc, vc) in zip(self.layers, self.kv):
            k, v = L.self_attn.ctx_kv(ctx, pos0)
            if kc is not None:
                k = mx.concatenate([kc, k], axis=2); v = mx.concatenate([vc, v], axis=2)
            if k.shape[2] > self.window:
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            new.append((k, v))
        self.kv = new
        self.ctx_end = pos0 + taps.shape[0]
        # Queued, not waited on: the projection runs while the next round's draft / verify graphs are built (the
        # round's single read-back follows the verify). A blocking eval here stalled the host once per round.
        mx.async_eval([x for kv in self.kv for x in kv])

    def draft(self, block_emb: mx.array, pos0: int) -> mx.array:
        """block_emb (1, B, D) at positions pos0.. (pos0 == ctx_end) -> final hidden (1, B, D)."""
        assert pos0 == self.ctx_end, (pos0, self.ctx_end)
        B = block_emb.shape[1]
        n = self.kv[0][0].shape[2] if self.kv[0][0] is not None else 0
        kpos = np.concatenate([np.arange(pos0 - n, pos0), np.arange(pos0, pos0 + B)])
        qpos = np.arange(pos0, pos0 + B)
        ok = kpos[None, :] > qpos[:, None] - self.window                  # non-causal, left window
        mask = mx.array(ok)[None, None]
        h = block_emb
        for L, (kc, vc) in zip(self.layers, self.kv):
            h = h + L.self_attn(L.input_layernorm(h), kc, vc, pos0, mask)
            h = h + L.mlp(L.post_attention_layernorm(h))
        return self.norm(h)


def load_drafter(path: str | Path) -> tuple[MiMoDFlashDrafter, mx.array | None]:
    path = Path(path)
    c = json.loads((path / "config.json").read_text())
    m = MiMoDFlashDrafter(c)
    w = {}
    for f in sorted(path.glob("*.safetensors")):
        if f.name.startswith("mask_embedding"):
            continue
        w.update(mx.load(str(f)))
    q = c.get("quantization")
    if q:
        def pred(p, mod):
            ov = q.get(p)
            return ov if isinstance(ov, dict) else False
        nn.quantize(m, class_predicate=pred)
    m.load_weights(list(w.items()), strict=True)
    mask_vec = None
    me = path / "mask_embedding.safetensors"
    if me.exists():
        d = mx.load(str(me))
        if int(d["mask_token_id"].item()) != m.mask_id:
            raise ValueError("mask_embedding mask_token_id differs from dflash_config.mask_token_id")
        mask_vec = d["embedding"]
    # materialize EVERYTHING returned here: served lanes load on the loader thread and generate on a worker thread, and a
    # lazy mx.load array evaluated from another thread aborts ("There is no Stream(cpu, 1) in current thread")
    mx.eval(m.parameters(), *([mask_vec] if mask_vec is not None else []))
    return m, mask_vec


def quantize_drafter(m: MiMoDFlashDrafter, mode: str):
    """Runtime quantization for A/B (bf16 files): 'affine8' | 'mxfp8'."""
    if mode == "bf16":
        return
    kw = dict(group_size=64, bits=8) if mode == "affine8" else dict(group_size=32, bits=8, mode="mxfp8")
    nn.quantize(m, class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and mod.weight.shape[1] % 64 == 0 and kw)


# ------------------------------------------------------------------------------------------------------------ target
class Target:
    """Text model (v26 Model) with output taps on the drafter's layers and SWA/full rollback."""

    def __init__(self, model, layer_ids):
        self.model, self.layer_ids = model, list(layer_ids)
        self.store = {}
        self._orig = {}
        for i in self.layer_ids:
            L = model.model.layers[i]
            cls = type(L)
            self._orig[i] = cls

            store = self.store

            class Tap(cls):
                def __call__(self, *a, **k):
                    out = super().__call__(*a, **k)
                    store[self._tap_idx] = out
                    return out
            L.__class__ = Tap
            object.__setattr__(L, "_tap_idx", i)

    def close(self):
        for i, cls in self._orig.items():
            self.model.model.layers[i].__class__ = cls

    def embed(self, ids):
        return self.model.model.embed_tokens(ids)

    def forward(self, ids: list[int], cache, embeds: mx.array | None = None,
                last_only: bool = False) -> tuple[mx.array, mx.array]:
        """-> (logits (S, V) fp32, taps (S, n_tap * D)). ``embeds`` (1, S, D): media prefill (merged vision/audio
        features from the omni model's get_input_embeddings) instead of the token embedding lookup. ``last_only``
        (prompt prefill): the head on the last row only -- a 2048-row chunk otherwise projected every row through the
        152,576-word head (~1.3 TFLOP and a 1.25 GB fp32 temporary) to keep one row."""
        self.store.clear()
        h = self.model.model(ids if isinstance(ids, mx.array) else mx.array(ids)[None], cache, embeds)
        if last_only:
            h = h[:, -1:]
        lg = self.model.lm_head(h)[0].astype(mx.float32)
        taps = mx.concatenate([self.store[i][0] for i in self.layer_ids], axis=-1)
        return lg, taps

    @staticmethod
    def rollback(cache, r: int):
        if r <= 0:
            return
        from mlx_lm.models.cache import RotatingKVCache, KVCache
        for c in cache:
            if isinstance(c, RotatingKVCache):
                if c.keys.shape[2] != c._idx or c.keys.shape[2] < r:
                    raise RuntimeError("rotating cache not in concat order after a multi-row verify")
                c.keys = c.keys[..., :-r, :]; c.values = c.values[..., :-r, :]
                c.offset -= r; c._idx = c.keys.shape[2]
            elif isinstance(c, KVCache):
                c.trim(r)
            else:
                raise RuntimeError(f"unsupported cache {type(c)}")


# ------------------------------------------------------------------------------------------------------------ loops
def _draft_p_min(sampled: bool = False) -> float:
    """Drafter-confidence cut: a draft block ends where the running product of the drafter's own probabilities of its
    drafts falls below the threshold (verify rows past a low-confidence draft are mostly rejected and each costs
    ~5-10 ms on this model: round costs 37 / 41 / 46 / 51 / 56 / 60 ms for 2-7 rows at short context, 50 / 61 / 70 /
    80 / 91 / 103 / 111 ms for 2-8 rows at 32-36k). The cut depends only on the drafter's proposal, never on the
    target, so greedy output and the sampled distribution are unchanged; it costs one read-back of the cut position
    before the verify is built.
    Defaults from the 2026-10-18 same-state A/B (6 engines per arm, ~1.6 GHz): greedy rounds cut at 0.4 (code +6.5 %,
    7k-context decode +2.8 % over the uncut lane; prose within noise); sampled rounds are not cut (the cut measured
    -3 % / within noise at the bundle's temperature 1.0). VMLX_MIMO_DFLASH_DRAFT_PMIN (greedy) and
    VMLX_MIMO_DFLASH_DRAFT_PMIN_SAMPLED take a value in (0, 1]; 0 turns the cut off."""
    import os

    name, default = (("VMLX_MIMO_DFLASH_DRAFT_PMIN_SAMPLED", "0") if sampled else ("VMLX_MIMO_DFLASH_DRAFT_PMIN", "0.4"))
    raw = os.environ.get(name)
    try:
        v = float(default if raw is None or raw.strip() == "" else raw)
    except ValueError:
        v = float(default)
    return v if 0.0 < v <= 1.0 else 0.0


def _confident_drafts(dl: mx.array, d_tok: mx.array, d_prob, p_min: float) -> int:
    """Number of leading drafts whose running drafter-probability product stays >= p_min. dl (1, n, V) fp32 drafter
    logits; d_prob (1, n, V) the sampling distribution the drafts were drawn from (None for greedy: the softmax)."""
    if int(d_tok.shape[1]) == 0:
        return 0
    if d_prob is None:
        q = mx.take_along_axis(mx.softmax(dl[0], axis=-1), d_tok[0][:, None], axis=-1)[:, 0]
    else:
        q = mx.take_along_axis(d_prob[0], d_tok[0][:, None], axis=-1)[:, 0]
    return int(mx.sum(mx.cumprod(q, axis=-1) >= p_min).item())


@dataclass
class Stats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    tokens: int = 0
    decode_s: float = 0.0
    accept_hist: dict = field(default_factory=dict)

    def as_dict(self):
        return {"rounds": self.rounds, "tokens": self.tokens, "tok_per_round": round(self.tokens / max(self.rounds, 1), 3),
                "acceptance": round(self.accepted / max(self.drafted, 1), 4), "decode_s": round(self.decode_s, 3),
                "decode_tok_s": round(self.tokens / self.decode_s, 2) if self.decode_s else None,
                "accept_hist": dict(sorted(self.accept_hist.items()))}


def _pick(lg, temp, key):
    if temp <= 0:
        return mx.argmax(lg, axis=-1)
    return mx.random.categorical(lg / temp, key=key)


def spec_generate(target: Target, drafter: MiMoDFlashDrafter, mask_vec, prompt: list[int], max_tokens: int, eos: set,
                  temp: float = 0.0, seed: int = 0, block: int | None = None):
    B = block or drafter.block
    cache = target.model.make_cache()
    drafter.reset()
    key = mx.random.key(seed)
    lg, taps = target.forward(prompt, cache)
    drafter.add_context(taps, 0)
    key, k1 = mx.random.split(key)
    anchor = int(_pick(lg[-1], temp, k1).item())
    out, st, pos = [anchor], Stats(), len(prompt)
    t0 = time.perf_counter()
    while len(out) < max_tokens and anchor not in eos:
        ids = mx.array([anchor] + [drafter.mask_id] * (B - 1))
        emb = target.embed(ids)[None]
        if mask_vec is not None:
            emb = mx.concatenate([emb[:, :1], mx.broadcast_to(mask_vec.astype(emb.dtype), (1, B - 1, emb.shape[-1]))], axis=1)
        dh = drafter.draft(emb, pos)
        dlg = target.model.lm_head(dh)[0, 1:].astype(mx.float32)          # drafter output is already final-normed
        drafts = [int(x) for x in np.asarray(mx.argmax(dlg, axis=-1))]
        lg, taps = target.forward([anchor] + drafts, cache)
        key, k1 = mx.random.split(key)
        picks = [int(x) for x in np.asarray(_pick(lg, temp, k1))]
        n = 0
        while n < len(drafts) and drafts[n] == picks[n]:
            n += 1
        new = drafts[:n] + [picks[n]]
        Target.rollback(cache, len(drafts) - n)
        drafter.add_context(taps[: n + 1], pos)                     # anchor + accepted drafts are now committed context
        pos += n + 1
        st.rounds += 1; st.drafted += len(drafts); st.accepted += n; st.accept_hist[n] = st.accept_hist.get(n, 0) + 1
        for t in new:
            out.append(t)
            if t in eos or len(out) >= max_tokens:
                break
        anchor = out[-1]
    mx.eval(mx.zeros(1)); st.decode_s = time.perf_counter() - t0; st.tokens = len(out) - 1
    return out, st


def ar_generate(target: Target, prompt: list[int], max_tokens: int, eos: set, temp: float = 0.0, seed: int = 0):
    cache = target.model.make_cache()
    key = mx.random.key(seed)
    lg, _ = target.forward(prompt, cache)
    key, k1 = mx.random.split(key)
    t = int(_pick(lg[-1], temp, k1).item()); out = [t]
    t0 = time.perf_counter()
    while len(out) < max_tokens and t not in eos:
        h = target.model.model(mx.array([[t]]), cache)
        lg = target.model.lm_head(h)[0, -1].astype(mx.float32)
        key, k1 = mx.random.split(key)
        t = int(_pick(lg, temp, k1).item()); out.append(t)
    dt = time.perf_counter() - t0
    return out, {"tokens": len(out) - 1, "decode_s": round(dt, 3), "decode_tok_s": round((len(out) - 1) / dt, 2) if dt else None}



# ------------------------------------------------------------------------------------------------------------ serving lane
_LANES: dict = {}


def _lane_for(model, drafter):
    key = (id(model), id(drafter))
    lane = _LANES.get(key)
    if lane is None:
        text = getattr(getattr(model, "language_model", None), "inner", None) or model
        lane = _LANES[key] = {"target": Target(text, drafter.layer_ids)}
    return lane


@dataclass
class LaneStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefill_s: float = 0.0
    rounds: int = 0
    ar_steps: int = 0
    accepted: list = field(default_factory=list)
    output_token_ids: list = field(default_factory=list)
    policy: dict = field(default_factory=dict)          # AdaptiveSpec's measured costs at the end (diagnostics)
    eos_forwarded: bool = False                          # a pipelined AR step forwarded the final EOS (cache holds it)
    cut_drafts: int = 0                                  # drafts dropped by the confidence cut (VMLX_MIMO_DFLASH_DRAFT_PMIN)


class _TapWindow:
    """The drafter's context rows (target taps) of the last ``window`` forwarded positions, appended per round and
    concatenated lazily (a round adds 1-8 rows; concatenating 1024 x 20480 every round would cost more than it saves)."""

    def __init__(self, window: int, rows: mx.array | None = None):
        self.window, self.parts, self.n = window, [], 0
        if rows is not None:
            self.add(rows)

    def add(self, rows: mx.array):
        self.parts.append(rows); self.n += int(rows.shape[0])
        if self.n > 2 * self.window:
            self.rows()

    def rows(self) -> mx.array | None:
        if not self.parts:
            return None
        out = self.parts[0] if len(self.parts) == 1 else mx.concatenate(self.parts, axis=0)
        out = out[-self.window:]
        self.parts, self.n = [out], int(out.shape[0])
        return out


def mimo_rounds(target: Target, drafter: MiMoDFlashDrafter, mask_vec, prompt_ids, max_tokens: int, eos: set,
                temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0, prefill_step: int = 2048,
                stats: LaneStats | None = None, adaptive: bool = True, block_size: int | None = None, controls=None,
                propose=None, resume: dict | None = None, snapshot_cuts=(), on_snapshot=None, session: dict | None = None,
                embeds: mx.array | None = None):
    """Yields (new_tokens, drafted) per cycle. Cache invariant: the last emitted token is sampled, not forwarded.
    Exact speculative sampling: drafts are DRAWN from the drafter's distribution (same temperature/top-p/top-k) and
    accepted by rejection sampling against the target's (dflash.model_mlx._rejection_sample); greedy = argmax match.

    Multi-turn reuse (vmlx_engine.mimo_session): ``resume`` = {"target_cache", "draft_context", "cache_len"} restored
    state, cache_len < len(prompt_ids); prefill stops exactly at each of ``snapshot_cuts`` once and calls
    ``on_snapshot(cut, cache, taps)``; ``session`` (out) receives the final "target_cache", "taps" (drafter context
    rows) and "forwarded" (positions the cache holds) when the generator ends, including when it is closed early.
    ``embeds`` (1, len(prompt_ids), D): media prefill embeddings, used for the positions this request prefills."""
    import dflash.model_mlx as runtime
    from dflash.model_mlx import _rejection_sample, _sampling_probs, _sample_probs
    from vmlx_engine.glm_dflash2 import AdaptiveSpec

    st = stats if stats is not None else LaneStats()
    W = drafter.window
    bs0 = int(block_size or drafter.block)
    t0 = time.perf_counter()
    prompt_ids = [int(t) for t in prompt_ids]
    if resume is not None:
        cache, reused = resume["target_cache"], int(resume["cache_len"])
        taps = resume["draft_context"]
        if taps is None or reused >= len(prompt_ids) or int(taps.shape[0]) != min(W, reused):
            raise ValueError("MiMo DFlash resume entry does not match its prompt / drafter window")
    else:
        cache, reused, taps = target.model.make_cache(), 0, None
    cuts = sorted({int(c) for c in snapshot_cuts if reused < int(c) < len(prompt_ids)})
    pos, logits = reused, None
    from vmlx_engine.utils.generation_cancellation import raise_if_generation_cancelled

    while pos < len(prompt_ids):
        # cooperative cancel at every prefill chunk boundary (SimpleEngine's scope): a client that dropped during a
        # 30k prompt otherwise held the engine until the whole prefill finished (follow-up waited 106-108 s, 2026-10-16)
        raise_if_generation_cancelled()
        end = min(len(prompt_ids), pos + prefill_step)
        end = next((c for c in cuts if pos < c < end), end)            # a snapshot is the cache exactly at its cut
        lg, tp = target.forward(prompt_ids[pos:end], cache, None if embeds is None else embeds[:, pos:end], last_only=True)
        logits = lg[-1:]
        taps = tp if taps is None else mx.concatenate([taps, tp], axis=0)
        taps = taps[-W:]
        mx.eval(logits, taps)
        pos = end
        if on_snapshot is not None and pos in cuts:
            on_snapshot(pos, cache, taps)
    st.prompt_tokens, st.cached_tokens = len(prompt_ids), reused
    drafter.reset(); drafter.ctx_end = len(prompt_ids) - taps.shape[0]; drafter.add_context(taps, drafter.ctx_end)
    window = _TapWindow(W, taps)
    fwd = len(prompt_ids)                                # tokens forwarded = drafter.ctx_end = cache positions
    try:
        yield from _decode_rounds(target, drafter, mask_vec, prompt_ids, logits, max_tokens, eos, temperature, top_p,
                                  top_k, st, adaptive, bs0, controls, propose, window, t0, runtime, _rejection_sample,
                                  _sampling_probs, _sample_probs, AdaptiveSpec, cache)
    finally:
        if session is not None:
            session["target_cache"], session["taps"] = cache, window.rows()
            # the target cache's own count (the drafter's context lags it inside an AR burst: deferred projection)
            session["forwarded"] = int(cache[0].offset)


def _decode_rounds(target, drafter, mask_vec, prompt_ids, logits, max_tokens, eos, temperature, top_p, top_k, st,
                   adaptive, bs0, controls, propose, window, t0, runtime, _rejection_sample, _sampling_probs,
                   _sample_probs, AdaptiveSpec, cache):
    fwd = len(prompt_ids)

    def proc(lg, hist, d_tok=None):                     # (S, V) -> (1, S, V), the requested TARGET distribution
        L = lg[None]
        return controls.process(L, hist, d_tok) if controls is not None else L

    def probs_of(L):
        return controls.probabilities(runtime, L, temperature, top_p, top_k) if controls is not None else \
            _sampling_probs(L, temperature, top_p, top_k)

    def pick(L):                                        # (1, S, V) -> (S,)
        return mx.argmax(L[0], axis=-1) if temperature <= 0 else _sample_probs(probs_of(L))[0]

    tokens = [int(pick(proc(logits[-1:], prompt_ids))[-1].item())]
    st.prefill_s = time.perf_counter() - t0
    st.output_token_ids = tokens
    yield [tokens[-1]], 0
    if tokens[-1] in eos:
        return
    from vmlx_engine.glm_dflash2 import adaptive_costs

    policy = AdaptiveSpec(bs0, costs=adaptive_costs(target, bs0)) if adaptive else None
    if policy is not None:
        # measured costs for the request's stats line: why a text kept drafting or fell back to AR
        st.policy = {"_policy": policy}
    n = 1                                                # fwd = tokens forwarded = cache positions
    after_ar = False
    p_min = _draft_p_min(sampled=temperature > 0) if propose is None else 0.0
    # AR bursts defer the drafter's context: their tap rows are projected in ONE add_context when the next draft round
    # starts (per-token projection of 5 drafter layers + an eval made every AR step slower than the plain lane's:
    # prose, where the policy rightly stops drafting, ran 39 tok/s on this lane vs 46 on the plain lane).
    pending, pending_at = [], fwd

    def _flush_pending():
        nonlocal pending, pending_at
        if pending:
            drafter.add_context(pending[0] if len(pending) == 1 else mx.concatenate(pending, axis=0), pending_at)
            pending = []
    while n < max_tokens:
        mode, b = policy.choose() if policy is not None else ("spec", bs0)
        if mode == "ar":
            after_ar = True; ta = time.perf_counter(); new = []
            k = min(b, max_tokens - n)

            def _row(tp_row):
                nonlocal pending_at, fwd
                if not pending:
                    pending_at = fwd
                pending.append(tp_row); window.add(tp_row); fwd += 1

            if controls is None:
                # Pipelined (as mlx_lm's generate loop): step i+1 is built from step i's lazily sampled token before
                # that token is read back, so graph construction overlaps the GPU (the synchronous burst measured
                # 23 ms/token here vs ~20 on the plain lane). A token that turns out to be EOS leaves its launched
                # step in the cache: that step forwarded the EOS itself, so the cache holds prompt + every emitted
                # token -- the end-of-turn state the next prompt (which re-renders that EOS) resumes after.
                lg, tp = target.forward(mx.array([[tokens[-1]]]), cache)
                y = pick(lg[None, -1:])
                mx.async_eval(y, tp)
                for i in range(k):
                    _row(tp)
                    nxt = None
                    if i + 1 < k:
                        lg2, tp2 = target.forward(y.reshape(1, 1), cache)
                        y2 = pick(lg2[None, -1:])
                        mx.async_eval(y2, tp2)
                        nxt = (y2, tp2)
                    t = int(y.item()); new.append(t); tokens.append(t)
                    if t in eos:
                        if nxt is not None:
                            _row(nxt[1]); st.eos_forwarded = True
                        break
                    if nxt is not None:
                        y, tp = nxt
            else:
                for _ in range(k):
                    lg, tp = target.forward([tokens[-1]], cache)
                    _row(tp)
                    t = int(pick(proc(lg[-1:], prompt_ids + tokens))[-1].item()); new.append(t); tokens.append(t)
                    if t in eos:
                        break
            policy.update_ar((time.perf_counter() - ta) / max(len(new), 1), len(new))
            st.ar_steps += len(new); n += len(new)
            yield new, 0
            if new[-1] in eos:
                return
            continue
        bs = min(b, max_tokens - n + 1)
        if bs <= 1:
            break
        _flush_pending()
        tr = time.perf_counter()
        emb = target.embed(mx.array([tokens[-1]] + [drafter.mask_id] * (bs - 1)))[None]
        if mask_vec is not None:
            emb = mx.concatenate([emb[:, :1], mx.broadcast_to(mask_vec.astype(emb.dtype), (1, bs - 1, emb.shape[-1]))], axis=1)
        dl = target.model.lm_head(drafter.draft(emb, fwd))[:, 1:].astype(mx.float32)          # (1, bs-1, V)
        if propose is not None:                         # tests: externally chosen drafts (greedy only)
            d_prob, d_tok = None, mx.array([propose(len(tokens), bs - 1)])
        elif temperature > 0:
            d_prob = _sampling_probs(dl, temperature, top_p, top_k); d_tok = _sample_probs(d_prob)
        else:
            d_prob = None; d_tok = mx.argmax(dl, axis=-1)
        # The drafts stay on the GPU: the verify forward is built from the lazy draft tokens while the drafter runs,
        # and the round reads back once, after the verify is queued. Reading the drafts to the host first left the GPU
        # idle while Python built the 48-layer verify graph (one extra host round trip per round). Same token ids,
        # same arithmetic and the same random-draw order: outputs are unchanged.
        d_tok = d_tok.astype(mx.int32)
        if p_min > 0.0:
            keep_d = _confident_drafts(dl, d_tok, d_prob, p_min)
            if keep_d < int(d_tok.shape[1]):
                d_tok = d_tok[:, :keep_d]
                if d_prob is not None:
                    d_prob = d_prob[:, :keep_d]
                st.cut_drafts += bs - 1 - keep_d
                bs = keep_d + 1
        mx.async_eval(d_tok)
        lg, tp = target.forward(mx.concatenate([mx.array([[tokens[-1]]], dtype=mx.int32), d_tok], axis=1), cache)
        L = proc(lg, prompt_ids + tokens, d_tok)
        if temperature > 0:
            t_prob = probs_of(L)
            mx.async_eval(t_prob, tp)
            drafts = d_tok[0].tolist()
            accepted, bonus = _rejection_sample(d_tok, t_prob, d_prob)
            bonus = int(bonus)
        else:
            t_tok = mx.argmax(L[0], axis=-1)
            mx.async_eval(t_tok, tp)
            drafts = d_tok[0].tolist()
            picks = t_tok.tolist()
            accepted = next((i for i in range(len(drafts)) if drafts[i] != picks[i]), len(drafts)); bonus = picks[accepted]
        new = (drafts[:accepted] + [bonus])[: max_tokens - n]
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        if stop is not None:
            new = new[: stop + 1]
        # The cache keeps the anchor + every emitted token but the last (sampled, not forwarded). A round cut short
        # by max_tokens or an EOS inside the accepted drafts keeps fewer rows than accepted + 1; a stored session
        # entry must hold exactly the emitted tokens.
        keep = len(new)
        Target.rollback(cache, len(drafts) + 1 - keep)
        drafter.add_context(tp[:keep], fwd); window.add(tp[:keep]); fwd += keep
        st.rounds += 1; st.accepted.append(accepted)
        if policy is not None and bs >= policy.bmin:
            # a cut block is recorded at the width it verified (its time is that width's cost)
            policy.update_spec(bs, accepted, time.perf_counter() - tr, after_ar=after_ar)
        after_ar = False
        tokens.extend(new); n += len(new)
        yield new, bs - 1
        if stop is not None:
            return


def is_mimo_v2(model) -> bool:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None and not isinstance(cfg, dict) else (cfg or {}).get("model_type")
    return mt == "mimo_v2" or getattr(model, "model_type", None) == "mimo_v2"


def stream_mimo_dflash(model, tokenizer, draft, prompt: str, *, max_tokens: int, temperature: float, top_p: float = 1.0,
                       top_k: int = 0, stop=None, prompt_tokens=None, sampling_controls=None):
    """Server lane: yields dflash.model_mlx.GenerationResponse chunks (the DFlash2 lane's contract) for a MiMo-V2.6
    target; ``draft`` = (MiMoDFlashDrafter, mask_vec) from load_drafter."""
    from dflash.model_mlx import GenerationResponse
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    import os

    drafter, mask_vec = draft
    tw = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
    if prompt_tokens is None:
        prompt_tokens = tw.encode(prompt, add_special_tokens=False)
    eos = set(getattr(tw, "eos_token_ids", None) or ()) | {151643, 151645, 151672}
    detok = tw.detokenizer; detok.reset()
    if stop:
        from vmlx_engine.dflash2_runtime import _StopDetokenizer
        detok = _StopDetokenizer(detok, [stop] if isinstance(stop, str) else list(stop))
    lane = _lane_for(model, drafter)
    st = LaneStats(); tic = time.perf_counter(); n = 0
    # Session reuse (RAM + SSD) lives in the MLLM wrapper's lane (models/mllm.py _stream_mimo_dflash); this generic
    # DFlash2-runtime entry point serves a request from an empty cache.
    for new, drafted in mimo_rounds(lane["target"], drafter, mask_vec, list(prompt_tokens), int(max_tokens), eos,
                                    float(temperature), float(top_p), int(top_k), stats=st,
                                    adaptive=os.environ.get("VMLX_MIMO_DFLASH_ADAPTIVE", "1") != "0",
                                    controls=sampling_controls):
        if n == 0:
            tic = time.perf_counter()
        for t in new:
            if t not in eos:
                detok.add_token(t)
        n += len(new)
        finished = new[-1] in eos or n >= max_tokens or getattr(detok, "matched", False)
        reason = None
        if finished:
            detok.finalize(); reason = "stop" if (new[-1] in eos or getattr(detok, "matched", False)) else "length"
        r = GenerationResponse(detok.last_segment, list(new), len(new), st.prompt_tokens,
                               st.prompt_tokens / max(st.prefill_s, 1e-9), n, n / max(time.perf_counter() - tic, 1e-9),
                               mx.get_peak_memory() / 1e9, reason)
        r.drafted = int(drafted)
        r.cached_tokens = int(st.cached_tokens)
        r.prefill_usage = {"tokens": int(st.prompt_tokens) - int(st.cached_tokens), "seconds": float(st.prefill_s),
                           "scope": "model_prefill_and_prompt_state", "path": "mimo_dflash"}
        r.cache_detail = "mimo-dflash-prefix" if st.cached_tokens else ""
        if finished:
            r.output_token_ids = list(st.output_token_ids)
        yield r
        if finished:
            return

# ------------------------------------------------------------------------------------------------------------ bench
PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, with tests.",
    "agentic": "You are a coding agent. The test suite fails with `KeyError: 'user_id'` in auth/session.py line 88 after a refactor that renamed the field to `uid`. Explain the fix and show the patch.",
    "explain": "Explain how a B-tree keeps itself balanced during insertions, step by step.",
    "prose": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "zh": "请解释一下量子纠缠是什么，并举一个生活中的类比。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--drafter", default=None)
    ap.add_argument("--draft-quant", choices=("as-stored", "bf16", "affine8", "mxfp8"), default="as-stored")
    ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--block", type=int, default=0); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--thinking", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from vmlx_engine.models.mimo_v26_janght import load_mimo_v26_janght_text
    model, tok = load_mimo_v26_janght_text(a.bundle)
    dpath = a.drafter or str(Path(a.bundle) / "dflash")
    drafter, mask_vec = load_drafter(dpath)
    if a.draft_quant != "as-stored":
        quantize_drafter(drafter, a.draft_quant); mx.eval(drafter.parameters())
    target = Target(model, drafter.layer_ids)
    eos = set(json.loads((Path(a.bundle) / "generation_config.json").read_text())["eos_token_id"])
    res = {"bundle": a.bundle, "drafter": dpath, "draft_quant": a.draft_quant, "temp": a.temp, "block": a.block or drafter.block, "runs": []}
    for r in range(a.rounds):
        for name, p in PROMPTS.items():
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=a.thinking, tokenize=True)
            ids = [int(t) for t in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
            arm_order = ("ar", "spec") if r % 2 == 0 else ("spec", "ar")              # alternate (thermal / cache drift)
            rec = {"round": r, "prompt": name}
            for arm in arm_order:
                if arm == "ar":
                    o, s = ar_generate(target, ids, a.max_tokens, eos, a.temp, seed=r)
                    rec["ar"] = s; rec["ar_text"] = tok.decode(o[:80])
                else:
                    o, s = spec_generate(target, drafter, mask_vec, ids, a.max_tokens, eos, a.temp, seed=r, block=a.block or None)
                    rec["spec"] = s.as_dict(); rec["spec_text"] = tok.decode(o[:80])
            rec["speedup"] = round(rec["spec"]["decode_tok_s"] / rec["ar"]["decode_tok_s"], 3) if rec["ar"]["decode_tok_s"] else None
            print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_text")}), flush=True)
            res["runs"].append(rec)
    by = {}
    for rec in res["runs"]:
        by.setdefault(rec["prompt"], []).append(rec)
    res["median"] = {k: {"tok_per_round": float(np.median([x["spec"]["tok_per_round"] for x in v])),
                         "speedup": float(np.median([x["speedup"] for x in v])),
                         "ar_tok_s": float(np.median([x["ar"]["decode_tok_s"] for x in v])),
                         "spec_tok_s": float(np.median([x["spec"]["decode_tok_s"] for x in v]))} for k, v in by.items()}
    print(json.dumps(res["median"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    target.close()


if __name__ == "__main__":
    main()

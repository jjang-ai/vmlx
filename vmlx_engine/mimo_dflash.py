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
        mx.eval([x for kv in self.kv for x in kv])

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

    def forward(self, ids: list[int], cache) -> tuple[mx.array, mx.array]:
        """-> (logits (S, V) fp32, taps (S, n_tap * D))."""
        self.store.clear()
        h = self.model.model(mx.array(ids)[None], cache)
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
def _clone_cache(cache):
    """Independent copies of MiMo's KVCache / RotatingKVCache states (logical size; later appends never alias)."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    out = []
    for c in cache:
        if isinstance(c, RotatingKVCache):
            n = RotatingKVCache(c.max_size, c.keep)
            if c.keys is not None:
                n.keys, n.values = c.keys * 1, c.values * 1
            n.offset, n._idx = c.offset, c._idx
        elif isinstance(c, KVCache):
            n = KVCache()
            if c.keys is not None:
                n.keys, n.values = c.keys[..., : c.offset, :] * 1, c.values[..., : c.offset, :] * 1
            n.offset = c.offset
        else:
            raise RuntimeError(f"MiMo DFlash prefix store: unsupported cache {type(c)}")
        out.append(n)
    return out


class MiMoPrefixStore:
    """RAM prompt-prefix snapshots for the MiMo DFlash lane: target caches, the drafter's context rows (last
    `window` taps) and the last logits, keyed by prompt ids; a request whose prompt starts with a stored prompt restores
    it and prefills only the suffix (multi-turn chats). LRU over VMLX_MIMO_DFLASH_PREFIX_ENTRIES (default 2)."""

    def __init__(self, max_entries: int = 2):
        from collections import OrderedDict
        self.max_entries = max(1, int(max_entries)); self.entries = OrderedDict()
        self.hits = self.misses = 0

    def lookup(self, ids):
        best = None
        for key in self.entries:
            if len(key) <= len(ids) and (best is None or len(key) > len(best)) and tuple(ids[: len(key)]) == key:
                best = key
        if best is None:
            self.misses += 1
            return None
        self.entries.move_to_end(best); self.hits += 1
        return len(best), self.entries[best]

    def put(self, ids, cache, taps, logits):
        key = tuple(int(t) for t in ids)
        snap = (_clone_cache(cache), taps, logits)
        mx.eval([x for c in snap[0] for x in (c.keys, c.values) if x is not None], taps, logits)
        self.entries[key] = snap; self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)


_LANES: dict = {}


def _lane_for(model, drafter):
    key = (id(model), id(drafter))
    lane = _LANES.get(key)
    if lane is None:
        text = getattr(getattr(model, "language_model", None), "inner", None) or model
        lane = _LANES[key] = {"target": Target(text, drafter.layer_ids),
                              "store": MiMoPrefixStore(int(__import__("os").environ.get("VMLX_MIMO_DFLASH_PREFIX_ENTRIES", "2")))}
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


def mimo_rounds(target: Target, drafter: MiMoDFlashDrafter, mask_vec, prompt_ids, max_tokens: int, eos: set,
                temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0, prefill_step: int = 2048,
                stats: LaneStats | None = None, store: MiMoPrefixStore | None = None, adaptive: bool = True,
                block_size: int | None = None, controls=None, propose=None):
    """Yields (new_tokens, drafted) per cycle. Cache invariant: the last emitted token is sampled, not forwarded.
    Exact speculative sampling: drafts are DRAWN from the drafter's distribution (same temperature/top-p/top-k) and
    accepted by rejection sampling against the target's (dflash.model_mlx._rejection_sample); greedy = argmax match."""
    import dflash.model_mlx as runtime
    from dflash.model_mlx import _rejection_sample, _sampling_probs, _sample_probs
    from vmlx_engine.glm_dflash2 import AdaptiveSpec

    st = stats if stats is not None else LaneStats()
    W = drafter.window
    bs0 = int(block_size or drafter.block)
    t0 = time.perf_counter()
    prompt_ids = [int(t) for t in prompt_ids]
    hit = store.lookup(prompt_ids) if store is not None else None
    reused = 0
    if hit is not None:
        reused, (snap_cache, taps, logits) = hit
        cache = _clone_cache(snap_cache)
    else:
        cache, taps, logits = target.model.make_cache(), None, None
    pos = reused
    while pos < len(prompt_ids):
        end = min(len(prompt_ids), pos + prefill_step)
        lg, tp = target.forward(prompt_ids[pos:end], cache)
        logits = lg[-1:]
        taps = tp if taps is None else mx.concatenate([taps, tp], axis=0)
        taps = taps[-W:]
        mx.eval(logits, taps)
        pos = end
    if store is not None and reused < len(prompt_ids):
        store.put(prompt_ids, cache, taps, logits)
    st.prompt_tokens, st.cached_tokens = len(prompt_ids), reused
    drafter.reset(); drafter.ctx_end = len(prompt_ids) - taps.shape[0]; drafter.add_context(taps, drafter.ctx_end)

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
    policy = AdaptiveSpec(bs0) if adaptive else None
    n, fwd = 1, len(prompt_ids)                          # fwd = tokens forwarded = drafter.ctx_end
    after_ar = False
    while n < max_tokens:
        mode, b = policy.choose() if policy is not None else ("spec", bs0)
        if mode == "ar":
            after_ar = True; ta = time.perf_counter(); new = []
            for _ in range(min(b, max_tokens - n)):
                lg, tp = target.forward([tokens[-1]], cache)
                drafter.add_context(tp, fwd); fwd += 1
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
        drafts = [int(x) for x in np.asarray(d_tok[0])]
        lg, tp = target.forward([tokens[-1]] + drafts, cache)
        L = proc(lg, prompt_ids + tokens, d_tok)
        if temperature > 0:
            accepted, bonus = _rejection_sample(d_tok, probs_of(L), d_prob)
            bonus = int(bonus)
        else:
            picks = [int(x) for x in np.asarray(mx.argmax(L[0], axis=-1))]
            accepted = next((i for i in range(len(drafts)) if drafts[i] != picks[i]), len(drafts)); bonus = picks[accepted]
        Target.rollback(cache, len(drafts) - accepted)
        drafter.add_context(tp[: accepted + 1], fwd); fwd += accepted + 1
        new = (drafts[:accepted] + [bonus])[: max_tokens - n]
        st.rounds += 1; st.accepted.append(accepted)
        if policy is not None:
            policy.update_spec(bs, accepted, time.perf_counter() - tr, after_ar=after_ar)
        after_ar = False
        stop = next((i for i, t in enumerate(new) if t in eos), None)
        if stop is not None:
            new = new[: stop + 1]
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
    store = lane["store"] if os.environ.get("VMLX_MIMO_DFLASH_PREFIX_CACHE", "1") != "0" else None
    for new, drafted in mimo_rounds(lane["target"], drafter, mask_vec, list(prompt_tokens), int(max_tokens), eos,
                                    float(temperature), float(top_p), int(top_k), stats=st, store=store,
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

# SPDX-License-Identifier: Apache-2.0
"""Ripple-65B-Preview (ricursiveAI, model_type "ricursive") for MLX.

Semantics follow modeling_ricursive.py @ 240d5966:
  - plain RMSNorm: weight * x * rsqrt(mean(x^2) + eps), the variance in float32. eps = 1e-6 is NOT negligible here: residual
    variance falls to about 1e-5 on layers 4-10, so a norm that ignores eps is wrong.
  - per-head Q/K RMSNorm over head_dim, before RoPE; V has no norm
  - four norms per block: input + post_attention around attention, pre_feedforward + post_feedforward around the FFN
  - value residual: v_in = lerp(scaled layer-0 embedding, normed input, sigmoid(value_res_lambda)), so v_proj has its own input
  - gated attention: attention output * sigmoid(g_proj(normed input)), the sigmoid in float32
  - FFN clamps: gate limited above at 20, up to +/-20, before silu(gate) * up
  - layer_types: sliding_attention (window 1024 = current token + 1023 before it, RoPE) / full_attention (no RoPE)
  - embed_scale multiplies the embedding before anything reads it, as a float32 product rounded back; untied lm_head; no BOS added by the template

register() installs this module as mlx_lm.models.ricursive so mlx_lm.load and anything built on it can resolve the model.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
import mlx_lm.models.cache as _cache
from mlx_lm.models.cache import KVCache


class RotatingKVCache(_cache.RotatingKVCache):
    """RotatingKVCache that a verify rollback can cut past its size, and that keeps the window exact after the cut. Keeps the base class NAME on purpose: prefix-cache storage
    dispatches on the class name, and any other name sends the rotating layers down the positional path, which slices their physical rows as if they were absolute (wrong output
    on every cache hit once a prompt takes more than one prefill chunk).

    A forward of S rows retains max_size + S - 1 rows and a single row max_size, so after dropping n <= S rows the next forward still sees the whole window. The
    base class cannot continue from there: with fewer than max_size rows at an offset past max_size, its in-place path computes a negative buffer size, so those rows
    go through its concatenating path until the buffer is full again. is_trimmable stays False past the window: a deeper cut would shorten the window without any error.
    """

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1 and self.keys is not None and self.offset >= self.max_size and self.keys.shape[2] < self.max_size:
            return self._update_concat(keys, values)
        k, v = super().update_and_fetch(keys, values)
        if keys.shape[2] == 1 and self._idx < k.shape[2] and self.offset > self._idx:
            # A single-row step past the window writes into the ring and the base class hands back the ring in physical
            # order. Attend in temporal order instead (oldest first, as a multi-row verify row sees its window), so an AR
            # step and the verify row for the same token reduce over the same keys in the same order. One window copy per
            # layer per step.
            return self._temporal_order(k), self._temporal_order(v)
        return k, v

    # The SSD session tier may store this layer as a delta (only the rows added since its parent entry) because state
    # is always the window in temporal order ending at ``offset`` (dflash2_session_disk reads this flag).
    _vmlx_temporal_state = True

    def _window(self):
        """(keys, values, rows) of the state: views, never the buffer objects (a single-row update rewrites those by
        slice assignment for every holder, so a snapshot would change under its owner), in temporal order, and at most
        max_size rows. After a multi-row forward the buffer holds max_size + S - 1 rows, but no later forward reads
        more than the newest max_size; a ring past its window is unrolled (one window copy per snapshot). A restored
        window with _idx = max_size continues exactly like the live ring: the next single row overwrites the oldest row
        and attends in temporal order (update_and_fetch), a multi-row append keeps the newest max_size - 1."""
        k, v = self.keys, self.values
        if self.offset < k.shape[2]:
            return k[..., : self.offset, :], v[..., : self.offset, :], self.offset
        if self._idx != k.shape[2]:
            k, v = self._temporal_order(k), self._temporal_order(v)
        n = min(k.shape[2], self.max_size)
        return k[..., -n:, :], v[..., -n:, :], n

    @property
    def state(self):
        if self.keys is None:
            return _cache.RotatingKVCache.state.fget(self)
        k, v, _ = self._window()
        return k, v

    @state.setter
    def state(self, v):
        _cache.RotatingKVCache.state.fset(self, v)

    @property
    def meta_state(self):
        meta = tuple(_cache.RotatingKVCache.meta_state.fget(self))
        if self.keys is not None and self.offset >= self.keys.shape[2]:
            meta = meta[:3] + (str(min(self.keys.shape[2], self.max_size)),) + meta[4:]
        return meta

    @meta_state.setter
    def meta_state(self, v):
        _cache.RotatingKVCache.meta_state.fset(self, v)

    def rollback_recent(self, n: int) -> None:
        if n <= 0 or self.keys is None:
            return
        self.keys = self._temporal_order(self.keys)[..., :-n, :]
        self.values = self._temporal_order(self.values)[..., :-n, :]
        self.offset -= n
        self._idx = self.keys.shape[2]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "ricursive"
    hidden_size: int = 5120
    intermediate_size: int = 40960
    num_hidden_layers: int = 80
    num_attention_heads: int = 80
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 1024
    max_position_embeddings: int = 2097152
    layer_types: list = field(default_factory=list)
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    use_valres: bool = True
    ffn_clamp_threshold: Optional[float] = 20.0
    embed_scale: Optional[float] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(self.num_hidden_layers)]
        assert len(self.layer_types) == self.num_hidden_layers, "layer_types must name every layer"
        assert set(self.layer_types) <= {"sliding_attention", "full_attention"}, sorted(set(self.layer_types))
        assert not self.tie_word_embeddings, "Ripple has an untied lm_head"


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        f = x.astype(mx.float32)
        y = f * mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + self.eps)
        return self.weight * y.astype(x.dtype)


def lerp(start: mx.array, end: mx.array, weight: mx.array) -> mx.array:
    return start + weight * (end - start)


def scale_embedding(h: mx.array, scale: Optional[float]) -> mx.array:
    """The product must be taken in float32 and then rounded. `h * scale` on a BF16 array rounds the scale first (71.55 -> 71.5), which differs from the
    reference on 14% of elements. Measured: float32-then-round is bit-exact against the reference on every element."""
    return h if scale is None else (h.astype(mx.float32) * scale).astype(h.dtype)


def _row_exact_window(rows: int) -> bool:
    """Inside a speculative verify that asked for decode arithmetic (vmlx_engine row_exact_verify_scope) and within its
    row coverage. Outside vMLX (no scope module) never."""
    import os

    if os.environ.get("VMLX_RICURSIVE_EXACT_ATTENTION", "1").strip().lower() in ("0", "off", "false", "no"):
        return False                                    # A/B: the verify window as one multi-row attention call
    try:
        from vmlx_engine.metal.row_exact_qmv import row_exact_scope_active, row_exact_scope_rows
    except ImportError:
        return False
    return row_exact_scope_active() and rows <= row_exact_scope_rows()


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.scale = args.head_dim ** -0.5
        self.window = args.sliding_window
        self.sliding = args.layer_types[layer] == "sliding_attention"
        self.gate_full_precision = args.gate_full_precision
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
        if args.use_gated_attention:
            self.g_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))
        self.q_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta) if self.sliding else None

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return self.rope(x.astype(mx.float32), offset=offset).astype(x.dtype)     # upstream applies RoPE in float32

    def value_states(self, x: mx.array, value_residual: Optional[mx.array]) -> mx.array:
        if value_residual is None:
            return x
        blend = mx.sigmoid(self.value_res_lambda[0].astype(mx.float32)).astype(x.dtype)
        return lerp(value_residual.astype(x.dtype), x, blend)

    def attend(self, q, k, v, mask, chunk: Optional[int]) -> mx.array:
        if chunk is None or q.shape[2] <= chunk:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        # Long rows, no cache: one query chunk at a time. A full 30k row would otherwise need an L x L map per head.
        T, parts = q.shape[2], []
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            k0 = max(0, s - (self.window - 1)) if self.sliding else 0
            qi = mx.arange(s, e)[:, None]; ki = mx.arange(k0, e)[None, :]
            m = ki <= qi
            if self.sliding:
                m = m & (ki >= qi - (self.window - 1))
            out = mx.fast.scaled_dot_product_attention(q[:, :, s:e], k[:, :, k0:e], v[:, :, k0:e], scale=self.scale, mask=m)
            mx.eval(out); parts.append(out)
        return mx.concatenate(parts, axis=2)

    def _exact_rows(self, q, k, v, cache):
        """A verify window whose every row equals its own decode step: RoPE per row at the row's offset, and each row
        attends alone over exactly the keys its decode step sees (sliding: the newest ``window`` keys ending at itself,
        in temporal order; full: every key up to itself, paired where MLX gives a pair the decode kernel)."""
        from vmlx_engine.metal.row_exact_qmv import attend_rows_in_pairs

        L = q.shape[2]
        if self.sliding:
            offset = cache.offset
            q = mx.concatenate([self._rope(q[:, :, r:r + 1], offset + r) for r in range(L)], axis=2)
            k = mx.concatenate([self._rope(k[:, :, r:r + 1], offset + r) for r in range(L)], axis=2)
        keys, values = cache.update_and_fetch(k, v)

        def sdpa(qq, kk, vv, m):
            return mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=self.scale, mask=m)

        if not self.sliding:
            return attend_rows_in_pairs(sdpa, q, keys, values)
        T, rows = keys.shape[2], []
        for r in range(L):
            end = T - L + r + 1
            lo = max(0, end - self.window)
            rows.append(sdpa(q[:, :, r:r + 1], keys[:, :, lo:end], values[:, :, lo:end], None))
        return mx.concatenate(rows, axis=2)

    def __call__(self, x, value_residual=None, mask=None, cache=None, chunk=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(self.value_states(x, value_residual))
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None and B == 1 and 1 < L and _row_exact_window(L):
            out = self._exact_rows(q, k, v, cache).transpose(0, 2, 1, 3).reshape(B, L, -1)
        else:
            if self.sliding:
                offset = cache.offset if cache is not None else 0
                q, k = self._rope(q, offset), self._rope(k, offset)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            out = self.attend(q, k, v, mask, chunk).transpose(0, 2, 1, 3).reshape(B, L, -1)
        if hasattr(self, "g_proj"):
            g = self.g_proj(x)
            if self.gate_full_precision:
                g = g.astype(mx.float32)
            out = out * mx.sigmoid(g).astype(out.dtype)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.clamp = args.ffn_clamp_threshold
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.clamp:
            gate = mx.minimum(gate, self.clamp)
            up = mx.clip(up, -self.clamp, self.clamp)
        return self.down_proj(nn.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.self_attn = Attention(args, layer)
        self.mlp = MLP(args)
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            setattr(self, n, RMSNorm(args.hidden_size, args.rms_norm_eps))

    def __call__(self, x, value_residual=None, mask=None, cache=None, chunk=None):
        a = self.self_attn(self.input_layernorm(x), value_residual, mask, cache, chunk)
        h = x + self.post_attention_layernorm(a)
        return h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))


class RicursiveModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_scale = args.embed_scale if args.embed_scale is not None else 1.0     # the drafter binds to it
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.ga_idx = args.layer_types.index("full_attention") if "full_attention" in args.layer_types else None
        self.swa_idx = args.layer_types.index("sliding_attention") if "sliding_attention" in args.layer_types else None

    def embed(self, inputs):
        return scale_embedding(self.embed_tokens(inputs), self.args.embed_scale)

    def masks(self, h, cache):
        full = create_attention_mask(h, cache[self.ga_idx]) if self.ga_idx is not None else None
        swa = (create_attention_mask(h, cache[self.swa_idx], window_size=self.args.sliding_window)
               if self.swa_idx is not None else None)
        return full, swa

    def __call__(self, inputs, cache=None, input_embeddings=None, return_unnormed=False, gdn_sink=None):
        """return_unnormed and gdn_sink are the drafter adapter's call shape; Ripple has no recurrent layer, so the sink stays empty."""
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full, swa = self.masks(h, cache)
        residual = h if self.args.use_valres else None
        for layer, c, kind in zip(self.layers, cache, self.args.layer_types):
            h = layer(h, residual, full if kind == "full_attention" else swa, c)
        return h if return_unnormed else self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RicursiveModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    @property
    def layers(self):
        return self.model.layers

    @property
    def language_model(self):
        """The drafter runtime reaches the text model through this name."""
        return self

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    def make_cache(self):
        return [KVCache() if kind == "full_attention" else RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                for kind in self.args.layer_types]


def register():
    """Make mlx_lm (and anything built on it) resolve model_type "ricursive" to this module."""
    sys.modules.setdefault("mlx_lm.models.ricursive", sys.modules[__name__])


def load(path, lazy: bool = False):
    register()
    from mlx_lm import load as _load
    return _load(str(path), lazy=lazy)

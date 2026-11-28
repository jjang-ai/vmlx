# SPDX-License-Identifier: Apache-2.0
"""Adaptive DFly2 speculative drafter & verification loop for Hy4-flash (`hy_v4_flash`).

Contract:
- Zero extra KV cache: each DFly2 layer `i` directly reuses target DSA layer `dfly_target_kv_layers[i]`
  (`[25, 29, 33, 37, 41]` for inline 5-layer DFly2) latent KV cache `(c_kv, k_pe)` AND its already-computed
  top-2048 sparse token indices — keeping draft cost O(2048) all the way to 1M context!
- Parallel block drafting (`W = K + 1` slots `[bonus_token, mask, ..., mask]`):
  1. `mtp_fusion` injects target's `last_hidden` into slot 0.
  2. All 5 dense MLA DFly2 layers run in parallel over `W` slots attending to `[target_topk_indices, block 0..W-1]`.
  3. `hidden_correction` runs sequentially for `step = 1..K`, conditioning on `embed_tokens(prev_tok)` before
     target's shared `lm_head`.
- Rule 0 (No Hard Draft Depth): `dfly_block_size = 6` (`K = 5`) only chooses the starting rung.
  Adaptive freely scales `K` up (`6, 7, 8, ...`) when full acceptance wins on measured wall time and
  down / to measured AR fallback (`ARFallback`) when acceptance drops.
- Lossless verification for BOTH greedy (`argmax`) and sampled (`min(1, p/q)` rejection sampling).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.dspark.generate import ARFallback, choose_width
from vmlx_engine.models.ernie5_5_moe.kernels import ernie_sparse_mla_prefill

from .cache import HyV4GrowingMLACache, HyV4LayerCache, rollback_hy4_caches, snapshot_hy4_caches
from .config import HyV4FlashConfig
from .language import HyV4FlashForCausalLM, _apply_interleaved_rope


@dataclass
class DFly2Stats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    ar_tokens: int = 0
    tokens: int = 0
    prefill_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    widths: Dict[int, int] = field(default_factory=dict)
    initial_draft_horizon: int = 0
    draft_horizon: int = 0
    horizon_probes: int = 0
    horizon_promotions: int = 0

    @property
    def tokens_per_round(self) -> float:
        return (self.tokens - 1 - self.ar_tokens) / self.rounds if self.rounds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "ar_tokens": self.ar_tokens,
            "tokens": self.tokens,
            "tokens_per_round": round(self.tokens_per_round, 3),
            "acceptance": round(self.acceptance, 4),
            "prefill_s": round(self.prefill_s, 3),
            "decode_s": round(self.decode_s, 3),
            "decode_tok_s": round((self.tokens - 1) / self.decode_s, 2) if self.decode_s > 0 else None,
            "widths": dict(sorted(self.widths.items())),
            "initial_draft_horizon": self.initial_draft_horizon,
            "draft_horizon": self.draft_horizon,
            "horizon_probes": self.horizon_probes,
            "horizon_promotions": self.horizon_promotions,
        }


class HyV4MTPFusion(nn.Module):
    """Slot-0 fusion injecting target post-norm hidden state into bonus token embedding."""

    def __init__(self, hidden_size: int, rms_eps: float = 1e-5) -> None:
        super().__init__()
        self.enorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gate = mx.ones((hidden_size,), dtype=mx.bfloat16)

    def __call__(self, embed_slot0: mx.array, target_hidden: mx.array) -> mx.array:
        cat = mx.concatenate([self.enorm(embed_slot0), self.hnorm(target_hidden)], axis=-1)
        return embed_slot0 + self.gate * self.eh_proj(cat)


class HyV4HiddenCorrection(nn.Module):
    """SwiGLU hidden-state correction conditioning draft slot `step` on previous token embedding."""

    def __init__(self, hidden_size: int, rms_eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.embed_norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.gate_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_slot: mx.array, prev_embed: mx.array) -> mx.array:
        cat = mx.concatenate([self.hidden_norm(hidden_slot), self.embed_norm(prev_embed)], axis=-1)
        return hidden_slot + self.down_proj(nn.silu(self.gate_proj(cat)) * self.up_proj(cat))


class HyV4DFly2Attention(nn.Module):
    """Gated MLA draft attention that directly reuses a paired target DSA layer's KV cache & sparse indices."""

    def __init__(self, config: HyV4FlashConfig) -> None:
        super().__init__()
        self.n_heads = int(config.num_attention_heads)  # 32
        self.q_lora_rank = int(config.q_lora_rank)  # 1024
        self.kv_lora_rank = int(config.kv_lora_rank)  # 512
        self.qk_nope = int(config.qk_nope_head_dim)  # 192
        self.qk_rope = int(config.qk_rope_head_dim)  # 64
        self.qk_head_dim = self.qk_nope + self.qk_rope  # 256
        self.v_head_dim = int(config.v_head_dim)  # 256
        self.scale = float(self.qk_head_dim ** -0.5)
        self.rope_theta = float(config.rope_theta_dsa)  # 1e7

        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj: Any = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = mx.zeros((self.n_heads, self.kv_lora_rank, self.qk_nope), dtype=mx.bfloat16)
        self.unembed_out = mx.zeros((self.n_heads, self.v_head_dim, self.kv_lora_rank), dtype=mx.bfloat16)
        self.linear_gate: Any = nn.Linear(config.hidden_size, self.n_heads * self.v_head_dim, bias=False)
        self.o_proj: Any = nn.Linear(self.n_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.learnable_sink_param = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.post_proj_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._sinks_bf16: Optional[mx.array] = None

    def set_kv_b_weight(self, kv_b_weight: mx.array) -> None:
        w = kv_b_weight.reshape(self.n_heads, self.qk_nope + self.v_head_dim, self.kv_lora_rank)
        k_part, v_part = mx.split(w, [self.qk_nope], axis=1)
        self.embed_q = mx.contiguous(k_part.swapaxes(-1, -2))
        self.unembed_out = mx.contiguous(v_part)

    def __call__(
        self,
        x: mx.array,
        target_kv_cache: HyV4GrowingMLACache,
        anchor_pos: int,
    ) -> mx.array:
        """Parallel causal attention over `[target_reused_context, draft_block(0..W-1)]`."""
        B, W, _ = x.shape
        q_c = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_c).reshape(B, W, self.n_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope], axis=-1)

        kva = self.kv_a_proj_with_mqa(x)
        blk_kv, blk_pe = mx.split(kva, [self.kv_lora_rank], axis=-1)
        blk_kv = self.kv_a_layernorm(blk_kv)[:, None, :, :]  # [1, 1, W, 512]
        blk_pe = blk_pe[:, None, :, :]  # [1, 1, W, 64]

        q_pe = _apply_interleaved_rope(q_pe, self.qk_rope, self.rope_theta, anchor_pos)
        blk_pe = _apply_interleaved_rope(blk_pe, self.qk_rope, self.rope_theta, anchor_pos)

        ctx_kv, ctx_pe = target_kv_cache.fetch_kv()
        q_abs = q_nope @ self.embed_q.swapaxes(-1, -2)
        sinks = (
            self._sinks_bf16
            if (q_nope.dtype == mx.bfloat16 and self._sinks_bf16 is not None)
            else self.learnable_sink_param.astype(q_nope.dtype)
        )

        # Build context + causal block KV using target's already-selected top-2048 indices!
        topk_idx = target_kv_cache.last_topk_indices
        if ctx_kv is not None and topk_idx is not None and B == 1:
            # Reuse last committed row's top-2048 indices [K_ctx] + block slots [S_ctx .. S_ctx + W - 1]
            S_ctx = int(ctx_kv.shape[2])
            ctx_row = topk_idx[0, -1, :]  # [K_ctx] uint32 (0xFFFFFFFF sentinel for invalid)
            all_kv = mx.concatenate([ctx_kv, blk_kv], axis=2)
            all_pe = mx.concatenate([ctx_pe, blk_pe], axis=2)
            # Build causal [1, W, K_ctx + W] index table: slot w sees ctx_row + block[0..w]
            blk_ids = mx.arange(S_ctx, S_ctx + W, dtype=mx.uint32)
            w_idx = mx.arange(W, dtype=mx.uint32)
            blk_causal = mx.where(
                w_idx[None, :] <= w_idx[:, None],
                blk_ids[None, :],
                mx.array(0xFFFFFFFF, dtype=mx.uint32),
            )  # [W, W]
            ctx_rep = mx.broadcast_to(ctx_row[None, :], (W, int(ctx_row.shape[0])))
            full_indices = mx.concatenate([ctx_rep, blk_causal], axis=-1)[None, :, :]  # [1, W, K_ctx + W]
            out_lat = ernie_sparse_mla_prefill(
                q_abs, q_pe, all_kv, all_pe, full_indices, self.learnable_sink_param, self.scale
            )
        else:
            # Short context (S_ctx <= 2048): attend directly over full ctx + causal block
            if ctx_kv is not None:
                all_kv = mx.concatenate([ctx_kv, blk_kv], axis=2)
                all_pe = mx.concatenate([ctx_pe, blk_pe], axis=2)
                S_ctx = int(ctx_kv.shape[2])
            else:
                all_kv, all_pe = blk_kv, blk_pe
                S_ctx = 0
            S_tot = S_ctx + W
            pe_scores = (q_pe * self.scale) @ all_pe.swapaxes(-1, -2)
            q_pos = mx.arange(S_ctx, S_tot, dtype=mx.int32)[:, None]
            k_pos = mx.arange(S_tot, dtype=mx.int32)[None, :]
            pe_scores = mx.where((k_pos <= q_pos)[None, None], pe_scores, -mx.inf)
            out_lat = mx.fast.scaled_dot_product_attention(
                q_abs, all_kv, all_kv, scale=self.scale, mask=pe_scores, sinks=sinks
            )

        out = (out_lat @ self.unembed_out.swapaxes(-1, -2)).transpose(0, 2, 1, 3).reshape(
            B, W, self.n_heads * self.v_head_dim
        )
        gate = mx.sigmoid(self.linear_gate(x))
        return self.post_proj_norm(self.o_proj(out * gate))


class HyV4DFly2MLP(nn.Module):
    """Dense SwiGLU MLP for inline 5-layer DFly2 (`intermediate_size=9216`, no post-MLP norm)."""

    def __init__(self, hidden_size: int, intermediate_size: int = 9216) -> None:
        super().__init__()
        self.gate_proj: Any = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj: Any = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj: Any = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HyV4DFly2Layer(nn.Module):
    """Single inline DFly2 decoder layer (no iHC, reuses paired target DSA KV cache & sparse indices)."""

    def __init__(self, config: HyV4FlashConfig, intermediate_size: int = 9216) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HyV4DFly2Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HyV4DFly2MLP(config.hidden_size, intermediate_size=intermediate_size)

    def __call__(
        self,
        x: mx.array,
        target_kv_cache: HyV4GrowingMLACache,
        anchor_pos: int,
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), target_kv_cache, anchor_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class HyV4DFly2Drafter(nn.Module):
    """Inline 5-layer DFly2 speculative drafter for Hy4-flash (`0 B` extra KV cache)."""

    def __init__(
        self,
        config: HyV4FlashConfig,
        *,
        intermediate_size: int = 9216,
        has_confidence_head: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = int(config.dfly_num_hidden_layers or len(config.dfly_target_kv_layers) or 5)
        self.block_size = int(config.dfly_block_size or 6)
        self.mask_token_id = int(config.dfly_mask_token_id)
        self.target_kv_layers = list(config.dfly_target_kv_layers or [25, 29, 33, 37, 41][: self.num_layers])

        self.mtp_fusion = HyV4MTPFusion(config.hidden_size, config.rms_norm_eps)
        self.layers = [HyV4DFly2Layer(config, intermediate_size=intermediate_size) for _ in range(self.num_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_correction: Optional[HyV4HiddenCorrection] = (
            HyV4HiddenCorrection(config.hidden_size, config.rms_norm_eps)
            if config.dfly_enable_hidden_correction
            else None
        )
        self.confidence_head: Optional[nn.Linear] = (
            nn.Linear(config.hidden_size, 1, bias=True) if has_confidence_head else None
        )

    def draft_block(
        self,
        target: HyV4FlashForCausalLM,
        target_caches: Sequence[HyV4LayerCache],
        bonus_token: mx.array,
        target_last_hidden: mx.array,
        k: int,
        *,
        temperature: float = 0.0,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[mx.array]]:
        """Propose `k` draft tokens in parallel block + sequential hidden_correction.

        Args:
            target: Bound `HyV4FlashForCausalLM` (provides shared `embed_tokens` and `lm_head`).
            target_caches: Live 42-layer target cache list (read-only; DFly2 never mutates it!).
            bonus_token: Scalar int32 `mx.array` (last accepted/sampled target token).
            target_last_hidden: `[1, 1, 3072]` post-norm hidden state from target's source row.
            k: Number of draft tokens to propose (`k >= 1`, governed by Adaptive Rule 0).
            temperature: Sampling temperature (`0.0` = greedy).
            sampler: Optional callable mapping log-probabilities `[1, V]` -> sampled token id.

        Returns:
            `(draft_ids [k] int32, draft_probs [k, V] or None, confidences [k] float32 or None)`
        """
        k = max(1, int(k))
        anchor_pos = int(target_caches[self.target_kv_layers[0]].offset)

        # 1. Construct parallel query block [bonus_token, mask, ..., mask] (width W = k + 1)
        b_tok = bonus_token.reshape(1, 1).astype(mx.int32)
        masks = mx.full((1, k), self.mask_token_id, dtype=mx.int32)
        block_ids = mx.concatenate([b_tok, masks], axis=1)  # [1, k + 1]

        e_block = target.model.embed_tokens(block_ids)  # [1, k + 1, 3072]
        slot0 = self.mtp_fusion(e_block[:, :1, :], target_last_hidden.reshape(1, 1, -1))
        h = mx.concatenate([slot0, e_block[:, 1:, :]], axis=1)

        # 2. Run all DFly2 layers in parallel reusing target's paired DSA caches & top-2048 indices!
        for layer, t_idx in zip(self.layers, self.target_kv_layers):
            t_inner = target_caches[t_idx].inner
            assert isinstance(t_inner, HyV4GrowingMLACache)
            h = layer(h, t_inner, anchor_pos)
        h = self.final_norm(h)  # [1, k + 1, 3072]

        confs: Optional[mx.array] = None
        if self.confidence_head is not None:
            confs = mx.sigmoid(self.confidence_head(h[:, 1:, :]).astype(mx.float32)).reshape(k)

        # 3. Sequential hidden_correction + shared lm_head for step = 1..k
        prev_tok = b_tok
        draft_ids: List[mx.array] = []
        draft_probs: List[mx.array] = []
        use_sample = temperature > 0.0 or sampler is not None
        for step in range(1, k + 1):
            h_step = h[:, step : step + 1, :]
            if self.hidden_correction is not None:
                prev_emb = target.model.embed_tokens(prev_tok)
                h_step = self.hidden_correction(h_step, prev_emb)
            logits = target.compute_logits(h_step)[0, 0]  # [V] float32
            if not use_sample:
                nxt = mx.argmax(logits, axis=-1).astype(mx.int32)
            else:
                scaled = logits / max(float(temperature), 1e-6) if temperature > 0.0 else logits
                log_p = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
                p = mx.exp(log_p)
                if sampler is not None:
                    nxt = sampler(log_p[None, :]).reshape(()).astype(mx.int32)
                else:
                    nxt = mx.random.categorical(log_p[None, :]).reshape(()).astype(mx.int32)
                draft_probs.append(p)
            draft_ids.append(nxt.reshape(()))
            prev_tok = nxt.reshape(1, 1)

        return (
            mx.stack(draft_ids),
            mx.stack(draft_probs) if draft_probs else None,
            confs,
        )


class HyV4DFly2Target:
    """Target adapter pairing `HyV4FlashForCausalLM` with `iter_hy4_dfly2`."""

    max_verify_rows: Optional[int] = None

    def __init__(self, model: HyV4FlashForCausalLM) -> None:
        self.model = model

    def make_cache(self) -> List[HyV4LayerCache]:
        return self.model.make_cache()

    def prefill(
        self,
        prompt_ids: Sequence[int],
        cache: List[HyV4LayerCache],
        prefill_step: int = 1024,
        *,
        start_pos: int = 0,
        snapshot_cuts: Sequence[int] = (),
        on_snapshot: Optional[Callable[[int, List[HyV4LayerCache], Any], None]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Chunked prefill returning `(last_logits [V], last_post_norm_hidden [1, 1, D])`."""
        ids = mx.array([list(prompt_ids)], dtype=mx.int32)
        total = int(ids.shape[1])
        step = max(1, int(prefill_step))
        pos = max(0, int(start_pos))
        cuts = sorted(c for c in set(int(x) for x in snapshot_cuts) if pos < c < total)
        logits, last_h = None, None
        while pos < total:
            end = min(total, pos + step)
            for c in cuts:
                if pos < c < end:
                    end = c
                    break
            chunk = ids[:, pos:end]
            is_last = end >= total
            if is_last:
                logits, h_all = self.model.forward_with_hidden(chunk, cache, tail_only=True)
                last_h = h_all[:, -1:, :]
                mx.eval(logits, last_h)
            else:
                h_mid = self.model.model(chunk, cache=cache, tail_only=True)
                mx.eval(h_mid)
            pos = end
            if on_snapshot is not None and pos in cuts:
                on_snapshot(pos, cache, None)
        assert logits is not None and last_h is not None
        return logits[0, -1], last_h


def _rejection_sample_dfly2(
    draft_ids: mx.array,
    target_logits: mx.array,
    draft_probs: mx.array,
    temperature: float,
) -> Tuple[int, int]:
    """Exact lossless Leviathan/Chen rejection sampling for DFly2 (`[k]` drafts, `[k+1, V]` target logits)."""
    k = int(draft_ids.shape[0])
    inv_t = 1.0 / max(float(temperature), 1e-6)
    t_scaled = target_logits.astype(mx.float32) * inv_t
    t_logp = t_scaled - mx.logsumexp(t_scaled, axis=-1, keepdims=True)
    t_probs = mx.exp(t_logp)  # [k + 1, V]

    d_list = draft_ids.tolist()
    u = mx.random.uniform(shape=(k,)).tolist()
    mx.eval(t_probs, draft_probs)

    accepted = 0
    for i in range(k):
        tid = int(d_list[i])
        p_t = float(t_probs[i, tid].item())
        p_d = float(draft_probs[i, tid].item())
        ratio = 1.0 if p_d <= 0.0 else min(1.0, p_t / p_d)
        if u[i] < ratio:
            accepted += 1
        else:
            # Residual distribution: norm(max(0, p_t - p_d))
            res = mx.maximum(t_probs[i] - draft_probs[i], 0.0)
            s = mx.sum(res)
            if float(s.item()) > 1e-12:
                res = res / s
                bonus = int(mx.random.categorical(mx.log(mx.maximum(res, 1e-30))[None, :]).item())
            else:
                bonus = int(mx.random.categorical(t_logp[i : i + 1]).item())
            return accepted, bonus

    # All k drafts accepted -> sample bonus from position k!
    bonus = int(mx.random.categorical(t_logp[k : k + 1]).item())
    return accepted, bonus


def iter_hy4_dfly2(
    target: HyV4FlashForCausalLM,
    drafter: HyV4DFly2Drafter,
    prompt_ids: Sequence[int],
    max_new: int,
    eos: set,
    *,
    temperature: float = 0.0,
    prefill_step: int = 1024,
    stats: Optional[DFly2Stats] = None,
    ar_fallback: Optional[ARFallback] = None,
    fixed_width: Optional[int] = None,
    cost_per_row: float = 0.12,
    max_draft: Optional[int] = None,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    pick: Optional[Callable[[mx.array], mx.array]] = None,
    resume: Optional[dict] = None,
    snapshot_cuts: Sequence[int] = (),
    on_snapshot: Optional[Callable[[int, List[HyV4LayerCache], Any], None]] = None,
) -> Iterator[List[int]]:
    """Adaptive Rule-0 DFly2 speculative decoding loop for Hy4-flash.

    - Starts at `k = drafter.block_size - 1` (5 for B6), NEVER hard-capped: winning full-accept
      boundary rounds promote the horizon upward (`6, 7, 8, ...`) while low-acceptance turns shrink `k`
      or switch smoothly to measured AR bursts (`ARFallback`).
    - Diagnostic env levers only: `VMLX_DFLY2_WIDTH=<k>` / `VMLX_DFLY2_MAX_DRAFT=<k>`.
    """
    st = stats if stats is not None else DFly2Stats()
    fb = ar_fallback if ar_fallback is not None else ARFallback(alpha=0.15, margin=0.03, probe_burst=2, ar_switch_streak=2)
    env_w = os.environ.get("VMLX_DFLY2_WIDTH", "").strip()
    env_max = os.environ.get("VMLX_DFLY2_MAX_DRAFT", "").strip()
    if fixed_width is None and env_w.isdigit() and int(env_w) >= 1:
        fixed_width = int(env_w)

    start_k = max(1, int(drafter.block_size) - 1)
    default_h = max(int(max_draft or 0), start_k * 2, 10)
    horizon = int(env_max) if (env_max.isdigit() and int(env_max) >= 1) else default_h
    st.initial_draft_horizon = st.draft_horizon = horizon

    adapter = HyV4DFly2Target(target)
    if resume and resume.get("target_cache") is not None:
        cache = resume["target_cache"]
        cache_len = int(resume.get("cache_len", 0) or 0)
    else:
        cache = adapter.make_cache()
        cache_len = 0

    def _pick_tok(lg_1d: mx.array) -> int:
        if pick is not None:
            return int(pick(lg_1d[None, :]).reshape(-1)[0].item())
        if temperature <= 0.0:
            return int(mx.argmax(lg_1d, axis=-1).item())
        sc = lg_1d.astype(mx.float32) / float(temperature)
        lp = sc - mx.logsumexp(sc, axis=-1, keepdims=True)
        return int(mx.random.categorical(lp[None, :]).item())

    t0 = time.perf_counter()
    first_logits, last_hidden = adapter.prefill(
        prompt_ids,
        cache,
        prefill_step=prefill_step,
        start_pos=cache_len,
        snapshot_cuts=snapshot_cuts,
        on_snapshot=on_snapshot,
    )
    st.prefill_tokens = len(prompt_ids) - cache_len
    st.prefill_s = time.perf_counter() - t0

    first_tok = _pick_tok(first_logits)
    emitted = 1
    st.tokens = 1
    yield [first_tok]
    if first_tok in eos or emitted >= max_new:
        return

    anchor = first_tok
    cur_k = fixed_width if fixed_width is not None else start_k
    t_dec0 = time.perf_counter()
    hasher = target.model._engram_hasher
    # Exact target-distribution verification:
    # - When caller supplies custom `pick` (top-p/min-p/temp sampler), sample-then-match draws every token
    #   from `pick(target_logits[i])` so output is 100% lossless with respect to any sampler!
    # - When `pick` is None and `temperature > 0`, use exact Leviathan/Chen rejection sampling.
    use_sample_then_match = pick is not None and sampler is not None
    use_rejection_sampling = pick is None and temperature > 0.0

    try:
        while emitted < max_new:
            rem = max_new - emitted

            # Measured AR burst when governor says AR beats drafting
            if fixed_width is None and fb.next_is_ar():
                burst = min(fb.burst_size(), rem)
                ta = time.perf_counter()
                ar_out: List[int] = []
                for _ in range(burst):
                    lg, h_all = target.forward_with_hidden(
                        mx.array([[anchor]], dtype=mx.int32), cache, tail_only=False
                    )
                    last_hidden = h_all[:, -1:, :]
                    nxt = _pick_tok(lg[0, -1])
                    mx.eval(last_hidden)
                    ar_out.append(nxt)
                    anchor = nxt
                    emitted += 1
                    st.ar_tokens += 1
                    st.tokens += 1
                    if nxt in eos or emitted >= max_new:
                        break
                fb.record_ar_burst(len(ar_out), time.perf_counter() - ta)
                yield ar_out
                if ar_out[-1] in eos:
                    break
                continue

            # Speculative DFly2 round
            k = min(cur_k if fixed_width is None else fixed_width, rem, horizon)
            if k < 1:
                break
            tr = time.perf_counter()
            d_ids, d_probs, confs = drafter.draft_block(
                target,
                cache,
                mx.array(anchor, dtype=mx.int32),
                last_hidden,
                k,
                temperature=temperature if use_rejection_sampling else 0.0,
                sampler=sampler if use_sample_then_match else None,
            )
            mx.async_eval(d_ids)

            # Snapshot caches BEFORE target verify so rejected suffix rolls back bit-exact!
            snap = snapshot_hy4_caches(cache)
            prev_eng_tail = snap[1]
            verify_ids = mx.concatenate([mx.array([anchor], dtype=mx.int32), d_ids], axis=0)[None, :]
            t_logits, t_h = target.forward_with_hidden(verify_ids, cache, tail_only=False)
            mx.eval(t_logits, t_h, d_ids)

            t_lg_2d = t_logits[0]  # [k + 1, V]
            if use_rejection_sampling:
                assert d_probs is not None
                accepted, bonus = _rejection_sample_dfly2(d_ids, t_lg_2d, d_probs, temperature)
            else:
                if use_sample_then_match:
                    t_picks = [int(pick(t_lg_2d[i : i + 1]).reshape(-1)[0].item()) for i in range(k + 1)]
                else:
                    t_picks = mx.argmax(t_lg_2d, axis=-1).tolist()
                d_list = d_ids.tolist()
                accepted = 0
                for i in range(k):
                    if int(d_list[i]) == int(t_picks[i]):
                        accepted += 1
                    else:
                        break
                bonus = int(t_picks[accepted])

            kept = accepted + 1  # anchor + accepted drafts committed into target cache!
            # Compute exact kept Engram compressed tail for rollback
            kept_tail = None
            if hasher is not None and kept > 0:
                v_np = np.array(verify_ids[0, :kept], dtype=np.int64)
                _, kept_tail, _ = hasher.hash_for_layers(v_np, prev_eng_tail)
            rollback_hy4_caches(cache, snap, kept=kept, kept_compressed_tail=kept_tail)
            last_hidden = t_h[:, accepted : accepted + 1, :]

            round_s = time.perf_counter() - tr
            new_toks = (d_ids[:accepted].tolist() + [bonus])[:rem]
            st.rounds += 1
            st.drafted += k
            st.accepted += accepted
            st.widths[k] = st.widths.get(k, 0) + 1
            fb.record_draft(len(new_toks), round_s, rows=k + 1)
            fb.record_acceptance(k, accepted)

            # Adaptive Rule-0 width governance (confidence + live acceptance + boundary horizon growth!)
            if fixed_width is None:
                if confs is not None:
                    mx.eval(confs)
                    c_list = confs.tolist()
                    if accepted == k and k >= horizon and not env_max:
                        horizon += 2
                        st.draft_horizon = horizon
                        st.horizon_probes += 1
                        st.horizon_promotions += 1
                    cur_k = max(1, min(horizon, fb.choose_width(c_list, horizon, cost_per_row)))
                    if accepted == k and cur_k == k and k < horizon:
                        cur_k = min(horizon, k + 1)
                else:
                    if accepted == k:
                        if k >= horizon and not env_max:
                            horizon += 2
                            st.draft_horizon = horizon
                            st.horizon_probes += 1
                            st.horizon_promotions += 1
                        cur_k = min(horizon, k + 1)
                    elif accepted <= max(0, k // 2 - 1):
                        cur_k = max(1, k - 1)

            stop_i = next((i for i, t in enumerate(new_toks) if t in eos), None)
            if stop_i is not None:
                new_toks = new_toks[: stop_i + 1]
            emitted += len(new_toks)
            st.tokens += len(new_toks)
            anchor = new_toks[-1]
            yield new_toks
            if stop_i is not None:
                break
    finally:
        st.decode_s = time.perf_counter() - t_dec0

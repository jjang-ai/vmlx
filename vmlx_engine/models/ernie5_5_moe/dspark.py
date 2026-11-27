# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 Flash bundled DSpark (JANGH3-q4fit) drafter and exact HCA/MLA rollback."""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from vmlx_engine.jangh.switch import TQSwitchLinear

from .config import ModelConfig
from .janght import (
    ErnieSwitchGLU,
    install_direct_mxfp8_linears,
    keep_fp8_linears,
    prepare_ernie_runtime,
)
from .kernels import ernie_dspark_markov_argmax, ernie_lm_head_forward
from .language import (
    DecoderLayer,
    ErnieRoPE,
    HyperHead,
    LanguageModel,
    MultiLinear,
    VirtualHeads,
)

logger = logging.getLogger(__name__)


class _DraftContextCache:
    """Thin trimmable wrapper around one DSpark draft layer's sliding context KV tensor."""

    def __init__(self, kv: Optional[mx.array], offset: int) -> None:
        self.kv = kv
        self.offset = int(offset)

    def trim(self, n: int) -> int:
        n = min(self.offset, max(0, int(n)))
        if n <= 0:
            return 0
        self.offset -= n
        if self.kv is not None:
            cur_len = int(self.kv.shape[2])
            if n >= cur_len:
                self.kv = None
            else:
                self.kv = mx.contiguous(self.kv[:, :, :-n, :])
        return n


class DraftAttention(nn.Module):
    """Non-causal sliding-window ERNIE DSpark draft attention with context KV."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim ** -0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a: Any = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=config.attention_bias)
        self.o_gate = nn.Linear(config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False)
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.rope = ErnieRoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )
        self._qb_ogate_fused: Optional[nn.Module] = None

    def context_kv(self, context: mx.array, offset: int) -> mx.array:
        return self.rope(self.kv_norm(self.wkv(context))[:, None], offset)

    def __call__(self, x: mx.array, attention_state: Tuple[mx.array, int]) -> mx.array:
        context_kv, offset = attention_state
        B, L, _ = x.shape
        qr = self.q_norm(self.wq_a(x))
        if self._qb_ogate_fused is not None:
            qb_og = self._qb_ogate_fused(qr)
            q_flat = qb_og[..., : self.n_heads * self.head_dim]
            ogate = qb_og[..., self.n_heads * self.head_dim :]
        else:
            q_flat = self.wq_b(qr)
            ogate = self.o_gate(qr)

        q = q_flat.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps).transpose(0, 2, 1, 3)
        q = self.rope(q, offset)
        kv = mx.concatenate((context_kv, self.context_kv(x, offset)), axis=2)
        queries = offset + mx.arange(L)
        keys = offset - context_kv.shape[2] + mx.arange(kv.shape[2])
        mask = keys[None] >= (queries[:, None] - self.config.sliding_window + 1)
        out = mx.fast.scaled_dot_product_attention(
            q, kv, kv, scale=self.scale, mask=mask, sinks=self.attn_sink.astype(q.dtype)
        )
        out = self.rope(out, offset, inverse=True).transpose(0, 2, 1, 3)
        out = self.vha(out).reshape(B, L, self.o_groups, -1).transpose(0, 2, 1, 3)
        out = self.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out * mx.sigmoid(ogate))


class ErnieDSparkDrafter:
    """Bundled ERNIE 5.5 DSpark JANGH3-q4fit non-causal drafter (iter_dspark compatible)."""

    def __init__(self, bundle_root: str | Path, target_config: ModelConfig) -> None:
        self.root = Path(bundle_root)
        self.dspark_dir = self.root / "dspark"
        manifest_path = self.dspark_dir / "draft-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ERNIE DSpark manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.manifest = manifest
        self.block_size = int(manifest["block_size"])
        self.block = self.block_size
        self.mask_id = int(manifest["mask_token"])
        self.mask_token_id = self.mask_id
        # ERNIE draft-manifest.json lists 1-based decoder layer ordinals (e.g. [10, 22, 34])
        self.taps: Tuple[int, ...] = tuple(int(x) for x in manifest["features"])
        self.target_layer_ids: List[int] = list(self.taps)
        n_layers = int(manifest["layers"])
        self.config = replace(
            target_config,
            num_hidden_layers=n_layers,
            first_k_dense_replace=0,
            compress_ratios=[128] * n_layers,
        )
        self.sliding_window = int(self.config.sliding_window)
        self.cfg = {
            "hidden_size": int(target_config.hidden_size),
            "vocab_size": int(target_config.vocab_size),
            "block_size": self.block_size,
            "num_hidden_layers": n_layers,
        }
        self._target: Optional[LanguageModel] = None
        self.layers: List[DecoderLayer] = []
        for idx in range(n_layers):
            block = DecoderLayer(self.config, idx)
            block.attn = DraftAttention(self.config, idx)
            raw = mx.load(str(self.dspark_dir / f"draft-layer-{idx}.safetensors"))
            prefix = f"model.layers.{idx}."
            switched = prefix + "ffn.switch_mlp."
            expert = {k: raw.pop(k) for k in tuple(raw) if k.startswith(switched)}

            keep_fp8_linears(prefix, raw)
            mapped = LanguageModel.sanitize_dict(self.config, raw)
            layer_w = {k.removeprefix(prefix): v for k, v in mapped.items()}
            install_direct_mxfp8_linears(block, layer_w)

            n_lat = self.config.moe_latent_size
            n_int = self.config.moe_intermediate_size
            n_exp = self.config.n_routed_experts
            projs = {}
            for pname, (din, dout) in (
                ("gate_proj", (n_lat, n_int)),
                ("up_proj", (n_lat, n_int)),
                ("down_proj", (n_int, n_lat)),
            ):
                tw = expert[switched + pname + ".tq2_packed"]
                bits = int(tw.shape[-1]) * 32 // din
                projs[pname] = TQSwitchLinear(din, dout, n_exp, bits, "hadamard32")
            block.ffn.switch_mlp = ErnieSwitchGLU(
                projs["gate_proj"], projs["up_proj"], projs["down_proj"],
                beta=self.config.activation_situ_beta,
                linear_beta=self.config.activation_situ_linear_beta,
            )
            for k, v in expert.items():
                layer_w[k.removeprefix(prefix)] = v
            block.load_weights(list(layer_w.items()), strict=True)
            mx.eval(block.parameters())
            block.eval()
            self.layers.append(block)
            del raw, mapped, layer_w, expert
            gc.collect()
            mx.clear_cache()

        prepare_ernie_runtime(self.layers)

        head = mx.load(str(self.dspark_dir / "draft-head.safetensors"))
        self.head = HyperHead(self.config)
        self.head.load_weights([
            (name, head[f"mhc_contract.hc_head_{name}"].T if name == "fn" else head[f"mhc_contract.hc_head_{name}"])
            for name in ("fn", "base", "scale")
        ])
        self.feature_norm = head["feature_norm.weight"]
        self.norm = head["norm.weight"]
        self.markov_embedding = mx.contiguous(head["markov_head.markov_w1.weight"].T)
        self.markov_head = head["markov_head.markov_w2.weight"]
        self.confidence = head["confidence_head.projection.weight"]
        self.feature_affine = {
            k.removeprefix("feature_affine4."): v
            for k, v in head.items()
            if k.startswith("feature_affine4.")
        }
        mx.eval(
            self.head.parameters(), self.feature_affine, self.feature_norm,
            self.norm, self.markov_embedding, self.markov_head, self.confidence,
        )
        # Per-session sliding context KV buffers ([1, 1, S_ctx, 512] per draft layer)
        self._ctx_kv: List[Optional[mx.array]] = [None] * len(self.layers)
        self._ctx_offset: int = 0
        self._ctx_base: int = 0
        logger.info(
            "ERNIE 5.5 DSpark drafter loaded: layers=%d block=%d taps=%s active=%.2f GiB",
            n_layers, self.block_size, self.taps, mx.get_active_memory() / 2**30,
        )

    def bind_target(self, target: LanguageModel) -> None:
        self._target = target

    # ------------------------------------------------------------------ iter_dspark context contract
    @property
    def ctx_len(self) -> int:
        return int(self._ctx_offset)

    @property
    def ctx_base(self) -> int:
        return int(self._ctx_base)

    def reset(self, base: int = 0) -> None:
        self._ctx_kv = [None] * len(self.layers)
        self._ctx_base = int(base)
        self._ctx_offset = int(base)

    def reset_context(self) -> None:
        self.reset(0)

    def set_context(self, caches: Sequence[_DraftContextCache], base: int = 0) -> None:
        self._ctx_base = int(base)
        self._ctx_kv = [c.kv for c in caches]
        self._ctx_offset = int(base) + (int(caches[0].offset) if caches else 0)

    def context_caches(self) -> List[_DraftContextCache]:
        rel = max(0, self._ctx_offset - self._ctx_base)
        return [_DraftContextCache(kv, rel) for kv in self._ctx_kv]

    def context_arrays(self) -> List[mx.array]:
        return [kv for kv in self._ctx_kv if kv is not None]

    def add_context(self, features: mx.array) -> None:
        self.append_features(features)

    def append_features(self, features: mx.array) -> None:
        """Project target tap features [1, T, 12288] -> context KV and keep last sliding_window entries."""
        if features is None or features.shape[1] == 0:
            return
        aff = self.feature_affine
        scaled = (features.astype(mx.float32) / aff["input_scale"]).astype(features.dtype)
        ctx = mx.quantized_matmul(
            scaled, aff["weight"], aff["scales"], aff["biases"],
            group_size=64, bits=4, mode="affine",
        )
        ctx = mx.fast.rms_norm(ctx, self.feature_norm, self.config.rms_norm_eps)
        win = self.config.sliding_window
        off = self._ctx_offset
        for i, block in enumerate(self.layers):
            new_kv = block.attn.context_kv(ctx, off)
            if self._ctx_kv[i] is None:
                merged = new_kv
            else:
                merged = mx.concatenate([self._ctx_kv[i], new_kv], axis=2)
            if merged.shape[2] > win:
                merged = mx.contiguous(merged[:, :, -win:, :])
            self._ctx_kv[i] = merged
        self._ctx_offset = off + int(features.shape[1])

    def _draft_block_logits(self, target: LanguageModel, anchor_arr: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        """Run the 3 non-causal draft layers and BF16 lm_head on the first `k` positions."""
        span = max(self.block_size, int(k))
        offset = self._ctx_offset
        win = self.config.sliding_window
        ctx_slices = [
            kv[:, :, -(min(kv.shape[2], max(0, win - 1))) :, :]
            if kv is not None
            else mx.zeros((1, 1, 0, self.config.head_dim), dtype=mx.bfloat16)
            for kv in self._ctx_kv
        ]
        mask_tail = mx.full((1, span - 1), self.mask_id, dtype=mx.int32)
        query = mx.concatenate([anchor_arr.reshape(1, 1).astype(mx.int32), mask_tail], axis=1)
        hidden = target.model.embed_tokens(query)
        hidden = mx.broadcast_to(hidden[:, :, None], (*hidden.shape[:2], self.config.hc_mult, hidden.shape[-1]))
        for block, kv in zip(self.layers, ctx_slices):
            hidden = block(hidden, (kv, offset))
        hidden = self.head(hidden)[:, :k, :]
        normed = mx.fast.rms_norm(hidden, self.norm, self.config.rms_norm_eps)
        base = ernie_lm_head_forward(
            normed, target.lm_head.weight,
            target.multimax.multimax_ranges, target.multimax.multimax_ts,
            apply_multimax=False,
        )[0]
        return hidden, base

    def draft_from_anchor(
        self,
        anchor: mx.array,
        k: int,
        sampler: Optional[Callable] = None,
    ) -> mx.array:
        """iter_dspark hook: propose `k` draft token ids (1D int32 mx.array) from anchor."""
        target = self._target
        if target is None:
            raise RuntimeError("ErnieDSparkDrafter has no bound LanguageModel target")
        k = max(1, int(k))
        _, base = self._draft_block_logits(target, anchor, k)
        token = anchor.reshape(()).astype(mx.int32)
        ranges = target.multimax.multimax_ranges
        ts = target.multimax.multimax_ts
        ids: List[mx.array] = []
        for i in range(k):
            transition = self.markov_embedding[token]
            if sampler is None:
                token = ernie_dspark_markov_argmax(base[i], self.markov_head, transition, ranges, ts)
            else:
                logits = base[i] + (self.markov_head.astype(mx.float32) @ transition.astype(mx.float32))
                logits = target.multimax(logits[None, :])[0]
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                token = sampler(lp[None, :]).reshape(()).astype(mx.int32)
            ids.append(token)
        return mx.stack(ids)

    def propose(
        self,
        target: LanguageModel,
        anchor: int,
        proposal_count: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Standalone helper: propose `count` greedy draft tokens + confidence logits."""
        count = self.block_size if proposal_count is None else max(1, int(proposal_count))
        anchor_arr = mx.array([anchor], dtype=mx.int32)
        hidden, base = self._draft_block_logits(target, anchor_arr, count)
        token = mx.array(anchor, dtype=mx.int32)
        ids: List[mx.array] = []
        confs: List[mx.array] = []
        ranges = target.multimax.multimax_ranges
        ts = target.multimax.multimax_ts
        for i in range(count):
            transition = self.markov_embedding[token]
            token = ernie_dspark_markov_argmax(base[i], self.markov_head, transition, ranges, ts)
            ids.append(token)
            confs.append(
                (self.confidence.astype(mx.float32) @ mx.concatenate((hidden[0, i], transition)).astype(mx.float32))[0]
            )
        ids_arr = mx.stack(ids)
        confs_arr = mx.stack(confs)
        mx.eval(ids_arr, confs_arr)
        return ids_arr.tolist(), confs_arr.tolist()


def snapshot_ernie_caches(caches: Sequence[Any]) -> List[Any]:
    """Lightweight O(1) snapshot of ERNIE HCA/MLA caches before a verify or carry AR step."""
    snaps = []
    for lc in caches:
        if lc.is_mla:
            snaps.append((lc[0].offset, lc[1].offset))
        else:
            local, pool = lc[0], lc[1]
            snaps.append((
                local.keys,
                local.offset,
                pool.pooled_count,
                pool.pending_kv,
                pool.pending_gate,
                pool.pending_len,
                pool.offset,
            ))
    return snaps


def rollback_ernie_caches(
    caches: Sequence[Any],
    snaps: Sequence[Any],
    kept: int,
    block_len: int,
) -> None:
    """Bit-exact rollback of ERNIE HCA/MLA caches keeping `kept` (0 <= kept <= block_len) verified tokens.

    - MLA layers: O(1) offset trim.
    - HCA local rotating cache: exact chronological tail slice + prepend from snapshot.
    - HCA pooling cache: O(1) pending slice trim, OR bit-exact restore + kept-slice append using
      `pool._last_in_kv` / `pool._last_in_gate` when a 128-token boundary fell inside the rejected suffix!
    """
    trim = int(block_len) - int(kept)
    if trim <= 0:
        return
    for lc, snap in zip(caches, snaps):
        if lc.is_mla:
            lc[0].trim(trim)
            lc[1].trim(trim)
            continue
        local, pool = lc[0], lc[1]
        prev_k, prev_off, prev_pc, prev_pkv, prev_pgate, prev_plen, prev_poff = snap

        # 1. Exact rollback of HCA local rotating cache (handles 128-boundary crossings bit-exact!)
        if kept == 0:
            local.keys = prev_k
            local.offset = prev_off
        elif local._last_in_keys is not None:
            kept_new = local._last_in_keys[:, :, :kept, :]
            if prev_k is None or prev_off == 0:
                merged = kept_new
            else:
                merged = mx.concatenate([prev_k, kept_new], axis=2)
            if merged.shape[2] > local.max_size:
                merged = merged[:, :, -local.max_size :, :]
            local.keys = mx.contiguous(merged)
            local.offset = prev_off + kept
        elif local.keys is not None:
            valid_from_new = local.keys[:, :, :-trim, :]
            local.keys = mx.contiguous(valid_from_new)
            local.offset = prev_off + kept
        else:
            local.offset = prev_off + kept

        # 2. Exact rollback of HCA pooling cache (handles kept == 0 AND 128-boundary crossings!)
        if kept == 0:
            pool.pooled_count = prev_pc
            pool.pending_kv = prev_pkv
            pool.pending_gate = prev_pgate
            pool.pending_len = prev_plen
            pool.offset = prev_poff
            continue

        r = pool.compress_ratio
        new_total_windows = (prev_poff + kept) // r
        if pool.pooled_count == new_total_windows and pool.pending_len >= trim:
            # Fast path (~96% of rounds): no rejected token completed a new 128-token window.
            new_plen = pool.pending_len - trim
            if new_plen > 0:
                pool.pending_kv = mx.contiguous(pool.pending_kv[:, :new_plen, :])
                pool.pending_gate = mx.contiguous(pool.pending_gate[:, :new_plen, :])
            else:
                pool.pending_kv = None
                pool.pending_gate = None
            pool.pending_len = new_plen
            pool.offset = prev_poff + kept
        else:
            # Exact boundary-crossing rollback using the tiny stashed verify input slice!
            in_kv = pool._last_in_kv
            in_gate = pool._last_in_gate
            pool.pooled_count = new_total_windows
            pool.offset = prev_poff + kept
            if in_kv is not None and in_gate is not None:
                # Reconstruct exact pending tail after `kept` tokens from (prev_pending + in[:kept]).
                if prev_pkv is not None and prev_plen > 0:
                    full_kv = mx.concatenate([prev_pkv, in_kv[:, :kept, :]], axis=1)
                    full_gate = mx.concatenate([prev_pgate, in_gate[:, :kept, :]], axis=1)
                else:
                    full_kv = in_kv[:, :kept, :]
                    full_gate = in_gate[:, :kept, :]
                consumed = (new_total_windows - prev_pc) * r
                rem = int(full_kv.shape[1]) - consumed
                if rem > 0:
                    pool.pending_kv = mx.contiguous(full_kv[:, consumed:, :])
                    pool.pending_gate = mx.contiguous(full_gate[:, consumed:, :])
                else:
                    pool.pending_kv = None
                    pool.pending_gate = None
                pool.pending_len = rem
            else:
                raise RuntimeError(
                    "ERNIE pooling cache boundary rollback missing verify input slice "
                    f"(prev_poff={prev_poff}, kept={kept}, block_len={block_len})"
                )

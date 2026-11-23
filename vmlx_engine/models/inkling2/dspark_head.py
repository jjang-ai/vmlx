# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's Inkling 2 DSpark implementation.
# Copyright contributors to the vLLM project.
"""Inkling 2's bundled BF16 DSpark with DFlare fusion and an RNN hidden head.

Reference: WoosukKwon/vllm@d664cd1e96052857aea2aa844eeba571c24758b4,
vllm/models/inkling2/{dspark,ops/dspark}.py (Apache-2.0).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.cache import KVCache, RotatingKVCache

from .dspark_config import DSparkConfig



class DraftAttention(nn.Module):
    def __init__(self, config, local):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window_size if local else None
        for name, heads in (
            ("q_proj", self.heads),
            ("k_proj", self.kv_heads),
            ("v_proj", self.kv_heads),
            ("k_proj_ctx", self.kv_heads),
            ("v_proj_ctx", self.kv_heads),
        ):
            self[name] = nn.Linear(
                config.hidden_size, heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(
            self.heads * self.head_dim, config.hidden_size, bias=False
        )
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            self[name] = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def _heads(self, x, heads):
        return x.reshape(*x.shape[:2], heads, self.head_dim).transpose(0, 2, 1, 3)

    def append_context(self, context, cache):
        offset = cache.offset
        k = self.k_norm_ctx(self._heads(self.k_proj_ctx(context), self.kv_heads))
        v = self._heads(self.v_proj_ctx(context), self.kv_heads)
        return cache.update_and_fetch(self.rope(k, offset=offset), v)

    def __call__(self, x, context, cache):
        # Every query sees the whole block and the same window BEFORE its anchor.
        # Only target context goes into persistent KV; proposals never do.
        offset = cache.offset
        k, v = self.append_context(context, cache)
        if isinstance(cache, RotatingKVCache):
            k, v = cache._temporal_order(k), cache._temporal_order(v)
            k, v = k[..., -self.window :, :], v[..., -self.window :, :]
        offset += context.shape[1]
        q = self.q_norm(self._heads(self.q_proj(x), self.heads))
        block_k = self.k_norm(self._heads(self.k_proj(x), self.kv_heads))
        k = mx.concatenate([k, self.rope(block_k, offset=offset)], axis=2)
        v = mx.concatenate([v, self._heads(self.v_proj(x), self.kv_heads)], axis=2)
        out = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset=offset), k, v, scale=self.head_dim**-0.5, mask=None
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))


class DraftMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        gate, up = self.gate_proj(x).astype(mx.float32), self.up_proj(x).astype(
            mx.float32
        )
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))


class DraftLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DraftAttention(config, index in config.local_layer_ids)
        self.mlp = DraftMLP(config)

    def __call__(self, x, residual, context, cache):
        # Fused add/RMSNorm normalizes the unrounded sum, but stores BF16 residuals.
        summed = (
            x
            if residual is None
            else x.astype(mx.float32) + residual.astype(mx.float32)
        )
        residual = summed.astype(x.dtype)
        x = self.self_attn(self.input_layernorm(summed).astype(x.dtype), context, cache)
        summed = x.astype(mx.float32) + residual.astype(mx.float32)
        return self.mlp(
            self.post_attention_layernorm(summed).astype(x.dtype)
        ), summed.astype(x.dtype)


class MarkovHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        rank, hidden = config.markov_rank, config.hidden_size
        self.markov_w1 = nn.Embedding(config.vocab_size, rank)
        self.rnn_proj = nn.Linear(2 * rank + 2 * hidden, 3 * rank)
        self.hidden_out = nn.Linear(rank, hidden)
        self.dflare_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self._fold_weight = self._fold_bias = None

    def fold(self, lm_head):
        # Match the reference's one-time FP64 folding, bounded to vocabulary tiles.
        projection = np.array(self.hidden_out.weight.astype(mx.float32)).astype(
            np.float64
        )
        bias = np.array(self.hidden_out.bias.astype(mx.float32)).astype(np.float64)
        weights, biases = [], []
        dtype = self.hidden_out.weight.dtype
        for start in range(0, lm_head.weight.shape[0], 4096):
            rows = slice(start, start + 4096)
            weight = lm_head.weight[rows]
            if isinstance(lm_head, nn.QuantizedLinear):
                weight = mx.dequantize(
                    weight,
                    lm_head.scales[rows],
                    lm_head.biases[rows] if "biases" in lm_head else None,
                    group_size=lm_head.group_size,
                    bits=lm_head.bits,
                    mode=lm_head.mode,
                )
            head = np.array(weight.astype(mx.float32)).astype(np.float64)
            weights.append(
                mx.array((head @ projection).astype(np.float32)).astype(dtype)
            )
            biases.append(mx.array((head @ bias).astype(np.float32)).astype(dtype))
        self._fold_weight, self._fold_bias = mx.concatenate(weights), mx.concatenate(
            biases
        )
        mx.eval(self._fold_weight, self._fold_bias)

    def context(self, hidden, f_rnn):
        rank = self.markov_w1.weight.shape[1]
        inputs = mx.concatenate([hidden, mx.broadcast_to(f_rnn, hidden.shape)], axis=-1)
        return inputs.astype(mx.float32) @ self.rnn_proj.weight[:, 2 * rank :].astype(
            mx.float32
        ).T + self.rnn_proj.bias.astype(mx.float32)

    def step(self, state, previous, context):
        rank = state.shape[-1]
        inputs = mx.concatenate([state, self.markov_w1(previous)], axis=-1)
        raw = (
            inputs.astype(mx.float32)
            @ self.rnn_proj.weight[:, : 2 * rank].astype(mx.float32).T
            + context
        ).astype(state.dtype)
        gate, candidate, output = mx.split(raw, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        candidate = mx.tanh(candidate.astype(mx.float32)).astype(state.dtype)
        state = gate * state + (1 - gate) * candidate
        latent = mx.tanh(output.astype(mx.float32)).astype(state.dtype)
        bias = mx.addmm(self._fold_bias, latent, self._fold_weight.T)
        return state, bias


class InklingDraft(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        self.layers = [DraftLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(config)
        self.fuse_weight = mx.zeros(
            (config.num_hidden_layers + 1, len(config.target_layer_ids))
        )
        self.mask_embed = mx.zeros((config.hidden_size,))
        self._target = None

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        args = getattr(target, "config", None)
        if (
            args is None
            or args.model_type not in ("inkling2_text", "inkling2_loop_text")
            or args.model_type != self.config.text_config.model_type
            or any(
                getattr(args, name) != getattr(self.config.text_config, name)
                for name in (
                    "hidden_size",
                    "vocab_size",
                    "num_hidden_layers",
                    "loop_config",
                )
            )
        ):
            raise ValueError("The DSpark head requires its matching Inkling 2 target")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        target = getattr(target_model, "language_model", target_model)
        if self._target is not target:
            self.markov_head.fold(target.lm_head)
            self._prepare_feature_mix()
            self._target = target
        return self

    def _prepare_feature_mix(self):
        if (
            self.config.hidden_size != 2304
            or self.fuse_weight.shape[1] != 4
            or self.fuse_weight.dtype != mx.bfloat16
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or mx.device_info().get("device_name") != "Apple M5 Max"
            or mx.__version__ not in ("0.32.4", "0.32.5.dev20261101+360078a8")
        ):
            return

        from .kernels import draft_feature_mix

        coefficients = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(mx.bfloat16)
        eps = mx.array([self.config.rms_norm_eps], mx.float32)
        # Compile the context-length-independent shader once before serving.
        probe = mx.zeros((1, 1, 4 * self.config.hidden_size), mx.bfloat16)
        mx.eval(
            draft_feature_mix(probe, coefficients[0], self.layers[0].dflare_norm.weight, eps)
        )
        self._fused_feature_mix = coefficients, eps

    def make_cache(self):
        return [
            (
                KVCache()
                if layer.self_attn.window is None
                else RotatingKVCache(max_size=layer.self_attn.window, keep=0)
            )
            for layer in self.layers
        ]

    def reset(self, target_model):
        self.bind(target_model)
        return self.make_cache()

    def mix(self, hidden, row, norm):
        fused = self.get("_fused_feature_mix")
        if (
            fused is not None and not self.training
            and hidden.shape[0] == 1 and hidden.shape[-1] == 4 * 2304
            and hidden.dtype == mx.bfloat16
            and norm.weight.dtype == mx.bfloat16
            and mx.default_device() == mx.gpu
        ):
            from .kernels import draft_feature_mix

            coefficients, eps = fused
            return draft_feature_mix(hidden, coefficients[row], norm.weight, eps)
        features = hidden.reshape(
            *hidden.shape[:2], -1, self.config.hidden_size
        ).astype(mx.float32)
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(
            hidden.dtype
        )[row]
        mixed = mx.zeros_like(features[:, :, 0])
        for i in range(features.shape[2]):
            mixed = mixed + features[:, :, i] * alpha[i].astype(mx.float32)
        return norm(mixed).astype(hidden.dtype)

    def _layer_context(self, hidden, index, layer, entry):
        window = layer.self_attn.window
        if window and hidden.shape[1] > window:
            # Old rows are never read by this layer. Keep its absolute RoPE offset.
            skipped = hidden.shape[1] - window
            hidden = hidden[:, skipped:]
            entry.offset += skipped
        return self.mix(hidden, index, layer.dflare_norm)

    def append_context(self, hidden, cache):
        for index, (layer, entry) in enumerate(zip(self.layers, cache)):
            layer.self_attn.append_context(
                self._layer_context(hidden, index, layer, entry), entry
            )

    def draft_block(
        self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32
    ):
        count = int(block_size) - 1
        if count < 0:
            raise ValueError("DSpark block must contain an anchor row")
        anchor = (
            mx.array([last_bonus], token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1)
        )
        if count == 0:
            return mx.zeros((anchor.size, 0), token_dtype)
        if self._target is None:
            raise RuntimeError("Bind the target before drafting")
        embed = self._target.model.embed(anchor[:, None])
        # Preserve the training block on narrower rounds. The requested span
        # can grow past that initial rung when the served controller observes a win.
        block_rows = max(count + 1, self.config.block_size)
        masks = mx.broadcast_to(
            self.mask_embed.astype(embed.dtype),
            (anchor.size, block_rows - 1, embed.shape[-1]),
        )
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            x, residual = layer(
                x, residual, self._layer_context(hidden, i, layer, entry), entry
            )
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(
            embed.dtype
        )[:, 1 : count + 1]
        # Draft logits are BF16, unlike the target's FP32 vocabulary projection.
        base = self._target.lm_head(x).astype(x.dtype)
        f_rnn = self.mix(hidden[:, -1:], -1, self.markov_head.dflare_norm)
        context = self.markov_head.context(x, f_rnn)
        state = mx.zeros((anchor.size, self.config.markov_rank), x.dtype)
        previous, tokens = anchor, []
        for i in range(count):
            state, bias = self.markov_head.step(state, previous, context[:, i])
            logits = base[:, i] + bias
            cap = self.config.final_logit_softcapping
            if cap:
                logits = (cap * mx.tanh(logits.astype(mx.float32) / cap)).astype(
                    x.dtype
                )
            previous = sampler(logits[:, : self.config.unpadded_vocab_size]).reshape(-1)
            tokens.append(previous[:, None])
        return mx.concatenate(tokens, axis=1).astype(token_dtype)

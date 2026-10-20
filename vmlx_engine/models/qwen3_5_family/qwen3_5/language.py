import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.gated_delta import gated_delta_update

from vmlx_engine.metal.quantized_projection_group import (
    QuantizedProjectionGroup,
    quantized_projection_group_reason,
)
from vmlx_engine.metal.gdn_conv_decode import (
    fused_qwen35_gdn_conv_requested,
    qwen35_gdn_conv_decode,
)
from vmlx_engine.metal.qwen35_gdn_gate_terms import (
    fused_qwen35_gdn_gates_requested,
    qwen35_gated_delta_decode,
)

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from .config import ModelConfig, TextConfig


_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def _unwrap_attention_cache(cache: Any) -> Any:
    """Return the cache behind vMLX's batch-offset compatibility proxy."""
    return getattr(cache, "_inner", cache)


def _qwen35_force_fused_prefill(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    mask: Optional[mx.array],
) -> bool:
    """Select MLX 0.32.2's low-memory D=256 full-attention kernel.

    MLX 0.32.2 automatically selects the NAX split-head kernel once the query
    span reaches 1,024 tokens. A non-divisor final chunk can be shorter than
    that while attending to a much longer cache, so the default heuristic
    falls back to materializing softmax(QK^T). Besides the memory cost, that
    fallback rounds differently from the fused one-shot lane. Force the fused
    kernel only for the exact ordinary-cache geometry MLX advertises.

    Quantized and TurboQuant caches retain their owning attention paths.
    Decode and short initial prompts retain MLX's latency-tuned heuristic.
    """
    enabled = (
        os.environ.get("VMLINUX_QWEN35_FUSED_PREFILL")
        or os.environ.get("VMLX_QWEN35_FUSED_PREFILL")
        or "1"
    )
    if enabled.strip().lower() in _FALSE_ENV_VALUES:
        return False

    raw_cache = _unwrap_attention_cache(cache)
    cache_name = type(raw_cache).__name__.lower()
    if hasattr(raw_cache, "bits") or "quant" in cache_name:
        return False

    query_length = int(queries.shape[-2])
    key_length = int(keys.shape[-2])
    return (
        isinstance(mask, str)
        and mask == "causal"
        and query_length > 8
        and key_length >= 1024
        and query_length <= key_length
        and int(queries.shape[-1]) == 256
        and int(values.shape[-1]) == 256
    )


def _qwen35_scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    if _qwen35_force_fused_prefill(queries, keys, values, cache, mask):
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            force_fused=True,
        )
    return scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
    )


class Qwen3_5RotaryEmbedding:
    def __init__(
        self, dim, max_position_embeddings=2048, base=10000, mrope_section=[11, 11, 0]
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

        self.mrope_section = mrope_section

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1):
    cos = mx.expand_dims(cos, axis=unqueeze_dim)
    sin = mx.expand_dims(sin, axis=unqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)

    return q_embed, k_embed


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            x = swiglu(gate, x)
        return x.astype(hidden_states.dtype)


class Qwen3_5Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rotary_emb = Qwen3_5RotaryEmbedding(
            int(self.head_dim * args.rope_parameters["partial_rotary_factor"]),
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_parameters["rope_theta"],
            mrope_section=args.rope_parameters["mrope_section"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]

        if position_ids is None:
            kv_seq_len += cache.offset + 1
            if position_embeddings is None:
                position_ids = mx.arange(cache.offset, cache.offset + L)
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        if position_embeddings is None:
            cos, sin = self.rotary_emb(values, position_ids)
        else:
            cos, sin = position_embeddings

        if mask is not None and isinstance(mask, mx.array):
            if isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = _qwen35_scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3_5MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.input_group = None
        self._fused_gdn_conv = fused_qwen35_gdn_conv_requested()
        self._fused_gdn_gates = fused_qwen35_gdn_gates_requested()

    def prepare_runtime(self) -> bool:
        """Group QKV/Z/B/A packed rows after checkpoint hydration."""

        linears = (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
        )
        if quantized_projection_group_reason(linears) is not None:
            return False
        group = QuantizedProjectionGroup(linears)
        mx.eval(group.weight, group.scales, group.biases)
        self.input_group = group
        self.in_proj_qkv = None
        self.in_proj_z = None
        self.in_proj_b = None
        self.in_proj_a = None
        return True

    def _project_inputs(
        self, inputs: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if self.input_group is not None:
            if inputs.ndim == 3 and inputs.shape[1] == 1:
                return self.input_group(inputs)
            return self.input_group.separate(inputs)
        return (
            self.in_proj_qkv(inputs),
            self.in_proj_z(inputs),
            self.in_proj_b(inputs),
            self.in_proj_a(inputs),
        )

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        mixed_qkv, z, b, a = self._project_inputs(inputs)
        z = z.reshape(B, S, -1, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        fused_conv = None
        if self._fused_gdn_conv and not self.training:
            fused_conv = qwen35_gdn_conv_decode(
                mixed_qkv,
                conv_state,
                self.conv1d.weight,
                enabled=True,
            )
        if fused_conv is not None:
            conv_out, next_conv_state = fused_conv
            if cache is not None:
                cache[0] = next_conv_state
        else:
            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            if cache is not None:
                # mx.contiguous, not a bare slice. A slice is a VIEW: it pins the
                # whole conv_input parent — (B, S+kernel-1, 8192) — for as long as
                # the cache entry lives, so ~34MB per layer is held to keep ~0.05MB
                # of state. Across 48 layers that is ~1.6GB of steady floor.
                #
                # It is a floor rather than a leak (one parent per layer, replaced
                # each chunk and released at the first decode step), which is why
                # the per-slot trace never showed it: `.nbytes` reports the VIEW
                # size, not the parent it holds alive.
                #
                # Every sibling capture site already does this — the MTP patch at
                # patches/mlx_lm_mtp/qwen35_model.py:270, the boundary captures in
                # mllm_batch_generator, and upstream mlx_lm. This vendored VLM copy
                # was the straggler, and it is the live path for Qwen3.5/3.6-VL
                # loads that go through mlx_vlm_compat without the MTP patch.
                cache[0] = mx.contiguous(
                    conv_input[:, -(self.conv_kernel_size - 1) :]
                )
            conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != B:
            state = None
        inv_scale = k.shape[-1] ** -0.5
        # Default: epsilon 1e-6 on the MEAN of squares (historical behaviour,
        # = d * 1e-6 on the sum). transformers/FLA l2norm adds 1e-6 to the
        # SUM; a model that sets ``qk_l2norm_eps`` (MiniCPM-V-4.7) gets that
        # exact form via eps / d on the mean.
        qk_eps = getattr(self, "qk_l2norm_eps", None)
        norm_eps = 1e-6 if qk_eps is None else float(qk_eps) / k.shape[-1]
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, norm_eps)
        k = inv_scale * mx.fast.rms_norm(k, None, norm_eps)

        # Opt-in speculative-verify capture (DSpark): the rows' recurrence
        # inputs, the state they started from and the raw conv window, so a
        # rejected suffix can be undone by replaying only the accepted rows
        # (``vmlx_engine.dspark.minicpm_target``). None by default.
        rollback_sink = getattr(self, "_rollback_sink", None)
        if rollback_sink is not None:
            rollback_sink.append(
                (q, k, v, a, b, state, mask,
                 mx.concatenate([conv_state, mixed_qkv], axis=1), self.conv_kernel_size)
            )

        fused_update = None
        if self._fused_gdn_gates and not self.training:
            fused_update = qwen35_gated_delta_decode(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                enabled=True,
            )
        if fused_update is not None:
            out, state = fused_update
        else:
            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )

        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3_5GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3_5Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3_5MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask,
                cache,
                position_ids,
                position_embeddings=position_embeddings,
            )
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3_5DecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def prepare_acceleration(self) -> dict[str, int]:
        """Install exact affine projection groups across base and MTP layers."""

        return prepare_quantized_projection_groups(self)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        # M-RoPE cos/sin are identical for every full-attention layer; for
        # media rows (3D position_ids) compute them once per forward instead
        # of once per layer. Text rows skip the hoist: the serve-path MTP
        # adapter routes them through a fused RoPE kernel that never reads
        # cos/sin, so hoisting there would be pure waste.
        position_embeddings = None
        if position_ids is not None and getattr(position_ids, "ndim", 0) == 3:
            fa_layer = self.layers[self.fa_idx]
            from vmlx_engine.metal.row_exact_qmv import (
                MAX_ROWS as _ROW_EXACT_MAX_ROWS, rotary_rows_like_decode, row_exact_scope_active,
            )

            if 1 < h.shape[1] <= _ROW_EXACT_MAX_ROWS and row_exact_scope_active():
                # Native-MTP verify rows: each angle as its decode step forms it.
                position_embeddings = rotary_rows_like_decode(
                    fa_layer.self_attn.rotary_emb, position_ids, h
                )
            else:
                position_embeddings = fa_layer.self_attn.rotary_emb(h, position_ids)

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(
                h, mask, c, position_ids, position_embeddings=position_embeddings
            )

        return self.norm(h)


def prepare_quantized_projection_groups(model: nn.Module) -> dict[str, int]:
    """Prepare exact Qwen3.5 groups without assuming dense vs MoE topology."""

    prepared = {
        "gdn_inputs": 0,
    }
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    seen: set[int] = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, Qwen3_5GatedDeltaNet) and module.prepare_runtime():
            prepared["gdn_inputs"] += 1
    mx.clear_cache()
    return prepared


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3_5Model(args)
        self._position_ids = None
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = mx.where(
                    attention_mask[i] == 1, input_ids, mx.zeros_like(input_ids)
                )
                image_nums, video_nums = 0, 0
                vision_start_indices = mx.sum(
                    mx.where(
                        input_ids == vision_start_token_id,
                        mx.arange(input_ids.shape[0]),
                        mx.zeros_like(input_ids),
                    )
                )
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
                    t_index = mx.broadcast_to(
                        t_index, (llm_grid_t, llm_grid_h * llm_grid_w)
                    )
                    t_index = t_index.flatten()

                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )
                    h_index = h_index.flatten()

                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )
                    w_index = w_index.flatten()

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st

                    t_index = mx.arange(text_len).reshape(1, text_len)
                    t_index = mx.broadcast_to(t_index, (3, text_len))

                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        new_positions,
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(
                    -1, keepdims=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        prefill_last_logits_only = bool(kwargs.pop("prefill_last_logits_only", False))
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        cache_offset = 0
        if cache and cache[self.model.fa_idx] is not None:
            offset = cache[self.model.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            elif offset is None:
                # A freshly constructed cache can report a None offset before the
                # first write; treat that as position 0 rather than crashing the
                # mRoPE delta path. (Defensive hardening formerly applied via
                # mlx_vlm_compat monkeypatch; kept here now the module is vendored.)
                cache_offset = 0
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        # Check if mask shape matches input shape (for chunked prefill compatibility)
        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            if (
                (
                    cache is not None
                    and cache[self.model.fa_idx] is not None
                    and (cache_offset == 0)
                )
                or self._rope_deltas is None
                or cache is None
            ):
                # Use cached position_ids when available (pre-computed in get_input_embeddings)
                if self._position_ids is not None:
                    seq_length = inputs.shape[1]
                    position_ids = self._position_ids[
                        :, :, cache_offset : cache_offset + seq_length
                    ]
                else:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
            else:
                batch_size, seq_length = inputs.shape
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)

                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        if prefill_last_logits_only and out.shape[1] > 1:
            # Prompt prefill: the caller samples only the last row. The head is
            # 248k x hidden, so projecting every prompt row cost ~5% of a cold
            # 8k-32k Ornith prefill and a (rows x vocab) transient. The trunk,
            # KV and recurrent state are untouched; the kept row goes through
            # the same one-row head kernel every decode step uses.
            out = out[:, -1:, :]
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def quant_predicate(self):

        if getattr(self.args, "num_experts", 0) <= 0:
            return None

        def predicate(path, _):
            # vMLX vendored fix: router gates MUST stay float. JANG affine
            # bundles (Ornith, JANG_4M/6M/1L) store gate weights as float16
            # without scales; forcing them through nn.quantize creates a
            # QuantizedLinear shell with a float16 packed weight, which then
            # crashes on quantized_matmul with:
            #   "The weight matrix should be uint32 but received float16"
            # mlx-lm's Qwen3NextSparseMoeBlock keeps gate as plain nn.Linear;
            # matching that here so both loader paths agree.
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return False
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


# Engine capability marker (mllm_batch_generator._lm_supports_prefill_last_logits): this exact __call__ honors
# prefill_last_logits_only. Subclasses that override __call__ do not inherit the marker.
LanguageModel.__call__._vmlx_prefill_last_logits_only = True

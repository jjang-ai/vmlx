# SPDX-License-Identifier: Apache-2.0
"""Native MTP adapter for mlx-vlm's Qwen3.5/3.6 language copy.

The text mlx-lm Qwen runtime and the mlx-vlm Qwen runtime are separate class
copies. This module installs the same concrete MTP contract on the VLM copy:
retain MTP config/weights, expose ``return_hidden``/``n_confirmed`` on the
language forward path, provide ``mtp_forward``/``make_mtp_cache``, and snapshot
hybrid SSM state for draft rejection rollback.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Apply the Qwen3.5/3.6 VLM native-MTP adapter. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_vlm.models.qwen3_5 import config as qcfg
        from mlx_vlm.models.qwen3_5 import language as qlang
        from mlx_vlm.models.qwen3_5 import qwen3_5 as qvl
    except ImportError as exc:
        logger.debug("mlx-vlm qwen3_5 not importable; skipping MTP adapter: %s", exc)
        return False

    try:
        from mlx_vlm.models.qwen3_5_moe import config as qmoe_cfg
        from mlx_vlm.models.qwen3_5_moe import language as qmoe_lang
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as qmoe_vl
    except ImportError:
        qmoe_cfg = None
        qmoe_lang = None
        qmoe_vl = None

    if (
        hasattr(qlang.LanguageModel, "mtp_forward")
        and not hasattr(qlang.LanguageModel, "_vmlx_mtp_patched")
    ):
        _PATCHED = True
        qlang.LanguageModel._vmlx_mtp_patched = "upstream"
        return True

    _patch_text_config(qcfg)
    _register_mtp_classes(qlang)
    _patch_attention_text_rope(qlang)
    _patch_gated_delta_net(qlang)
    _patch_decoder_layer(qlang)
    _patch_qwen_model(qlang)
    _patch_language_model(qlang)
    _patch_outer_model(qvl)

    if qmoe_lang is not None and qmoe_vl is not None:
        if qmoe_cfg is not None:
            _patch_text_config(qmoe_cfg)
        _register_moe_mtp_classes(qmoe_lang, qlang)
        _patch_moe_decoder_layer(qmoe_lang)
        _patch_moe_language_model(qmoe_lang)
        _patch_moe_outer_model(qmoe_vl)

    _PATCHED = True
    qlang.LanguageModel._vmlx_mtp_patched = "patch"
    logger.info("mlx-vlm Qwen3.5/3.6 MTP language adapter applied")
    return True


def _patch_text_config(qcfg: Any) -> None:
    cls = qcfg.TextConfig
    if hasattr(cls, "_vmlx_mtp_from_dict_patched"):
        return

    original_from_dict = cls.from_dict.__func__

    def patched_from_dict(config_cls, params):
        instance = original_from_dict(config_cls, params)
        raw_mtp_layers = None
        if isinstance(params, dict):
            raw_mtp_layers = params.get("mtp_num_hidden_layers")
            text_config = params.get("text_config")
            if raw_mtp_layers is None and isinstance(text_config, dict):
                raw_mtp_layers = text_config.get("mtp_num_hidden_layers")
        instance.mtp_num_hidden_layers = int(raw_mtp_layers or 0)
        return instance

    cls.from_dict = classmethod(patched_from_dict)
    cls._vmlx_mtp_from_dict_patched = True


def _mtp_layer_count(args: Any, config: Any = None) -> int:
    raw = getattr(args, "mtp_num_hidden_layers", None)
    if raw is None and config is not None:
        raw = getattr(getattr(config, "text_config", None), "mtp_num_hidden_layers", None)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _register_mtp_classes(qlang: Any) -> None:
    if hasattr(qlang, "MTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache

    class MTPDecoderLayer(nn.Module):
        """Full-attention Qwen MTP layer for the VLM language copy."""

        def __init__(self, args):
            super().__init__()
            self.self_attn = qlang.Qwen3_5Attention(args)
            self.input_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = qlang.Qwen3_5MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None, position_ids=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Qwen3.6 MTP head: fuse hidden(t) with sampled token(t+1)."""

        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                MTPDecoderLayer(args) for _ in range(_mtp_layer_count(args))
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [KVCache() for _ in self.layers]

            mask = qlang.create_attention_mask(fused, cache[0] if cache else None)
            for layer, layer_cache in zip(self.layers, cache):
                fused = layer(fused, mask, layer_cache)

            return self.norm(fused)

    qlang.MTPDecoderLayer = MTPDecoderLayer
    qlang.MTPModule = MTPModule


def _patch_attention_text_rope(qlang: Any) -> None:
    cls = qlang.Qwen3_5Attention
    if "_vmlx_text_rope_patched" in cls.__dict__:
        return

    original_call = cls.__call__

    def _text_rope(self):
        rope = getattr(self, "_vmlx_text_rope_kernel", None)
        if rope is None:
            rope = qlang.nn.RoPE(
                int(self.rotary_emb.dim),
                traditional=False,
                base=float(self.rotary_emb.base),
            )
            self._vmlx_text_rope_kernel = rope
        return rope

    def _position_offset(position_ids) -> int:
        try:
            first = position_ids[0, 0]
            return int(first.item() if hasattr(first, "item") else first)
        except Exception:
            return 0

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
    ):
        if position_ids is not None and getattr(position_ids, "ndim", 0) != 2:
            return original_call(self, x, mask, cache, position_ids)

        batch_size, seq_len, _ = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = qlang.mx.split(
            q_proj_output.reshape(batch_size, seq_len, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch_size, seq_len, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, seq_len, self.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        kv_seq_len = keys.shape[-2]
        text_position_ids = position_ids is not None and position_ids.ndim == 2
        if position_ids is None:
            offset = cache.offset if cache is not None else 0
            kv_seq_len += offset + 1
            text_position_ids = True
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0
            offset = _position_offset(position_ids)

        if text_position_ids:
            # mlx-vlm's M-RoPE implementation is required for true media
            # positions, but Qwen3.5/3.6 text-only axes must match mlx-lm's
            # 1D RoPE path. Use the native RoPE kernel for those rows instead
            # of passing identical text axes through the multimodal splitter.
            rope = self._vmlx_text_rope()
            queries = rope(queries, offset=offset)
            keys = rope(keys, offset=offset)
        else:
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = qlang.apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin
            )

        if mask is not None and isinstance(mask, qlang.mx.array):
            if isinstance(kv_seq_len, qlang.mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = qlang.scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.o_proj(output * qlang.mx.sigmoid(gate))

    cls._vmlx_text_rope = _text_rope
    cls.__call__ = __call__
    cls._vmlx_text_rope_patched = True


def _patch_gated_delta_net(qlang: Any) -> None:
    cls = qlang.Qwen3_5GatedDeltaNet
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    import mlx.core as mx
    import mlx.nn as nn

    def _process_chunk(
        self,
        qkv_chunk,
        a_chunk,
        b_chunk,
        conv_state,
        ssm_state,
        ssm_mask=None,
        lengths=None,
    ):
        batch_size, seq_len = qkv_chunk.shape[:2]
        conv_input = mx.concatenate([conv_state, qkv_chunk], axis=1)
        keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, seq_len)
            positions = (ends[:, None] + mx.arange(keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_input[:, -keep:])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(batch_size, seq_len, heads, dim)
            for t, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, new_ssm_state = qlang.gated_delta_update(
            q,
            k,
            v,
            a_chunk,
            b_chunk,
            self.A_log,
            self.dt_bias,
            ssm_state,
            ssm_mask,
            use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state

    def __call__(
        self,
        inputs: Any,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
    ):
        batch_size, seq_len, _ = inputs.shape

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch_size:
                conv_state = None
        else:
            conv_state = None
        if conv_state is None:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        ssm_state = cache[1] if cache else None
        if ssm_state is not None and ssm_state.shape[0] != batch_size:
            ssm_state = None

        if mask is not None:
            if mask.shape[0] != batch_size:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)

        if n_confirmed > 0 and n_confirmed < seq_len:
            mask_c = mask[:, :n_confirmed] if mask is not None else None
            mask_d = mask[:, n_confirmed:] if mask is not None else None
            out_c, conv_c, ssm_c = self._process_chunk(
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                mask_c,
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)
            out_d, conv_f, ssm_f = self._process_chunk(
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_c,
                ssm_c,
                mask_d,
            )
            out = mx.concatenate([out_c, out_d], axis=1)
        else:
            lengths = getattr(cache, "lengths", None) if cache is not None else None
            out, conv_f, ssm_f = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            advance = getattr(cache, "advance", None)
            if callable(advance):
                advance(seq_len)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch_size, seq_len, -1))

    cls._process_chunk = _process_chunk
    cls.__call__ = __call__
    cls._vmlx_mtp_patched = True


def _patch_decoder_layer(qlang: Any) -> None:
    cls = qlang.Qwen3_5DecoderLayer
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    original_call = cls.__call__

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        n_confirmed: int = 0,
    ):
        if n_confirmed == 0:
            return original_call(self, x, mask, cache, position_ids)

        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    cls.__call__ = __call__
    cls._vmlx_mtp_patched = True


def _patch_moe_decoder_layer(qmoe_lang: Any) -> None:
    cls = qmoe_lang.Qwen3_5MoeDecoderLayer
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    original_call = cls.__call__

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        n_confirmed: int = 0,
    ):
        if n_confirmed == 0:
            return original_call(self, x, mask, cache, position_ids)

        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    cls.__call__ = __call__
    cls._vmlx_mtp_patched = True


def _patch_qwen_model(qlang: Any) -> None:
    cls = qlang.Qwen3_5Model
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    original_call = cls.__call__

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        n_confirmed: int = 0,
        return_unnormed: bool = False,
    ):
        if n_confirmed == 0 and not return_unnormed:
            return original_call(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
            )

        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = qlang.create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = qlang.create_ssm_mask(h, cache[self.ssm_idx])

        for layer, layer_cache in zip(self.layers, cache):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(
                h,
                layer_mask,
                layer_cache,
                position_ids=position_ids,
                n_confirmed=n_confirmed,
            )

        return h

    cls.__call__ = __call__
    cls._vmlx_mtp_patched = True


def _register_moe_mtp_classes(qmoe_lang: Any, qlang: Any) -> None:
    if hasattr(qmoe_lang, "MTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache

    class MTPDecoderLayer(nn.Module):
        """Full-attention MoE MTP layer for Qwen3.5/3.6 VLM-MoE."""

        def __init__(self, args):
            super().__init__()
            self.self_attn = qmoe_lang.Qwen3_5MoeAttention(args)
            self.input_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = qmoe_lang.Qwen3_5MoeSparseMoeBlock(args)

        def __call__(self, x, mask=None, cache=None, position_ids=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Qwen3.6 MoE MTP head: hidden(t) + sampled token(t+1)."""

        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                MTPDecoderLayer(args) for _ in range(_mtp_layer_count(args))
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [KVCache() for _ in self.layers]

            mask = qlang.create_attention_mask(fused, cache[0] if cache else None)
            for layer, layer_cache in zip(self.layers, cache):
                fused = layer(fused, mask, layer_cache)

            return self.norm(fused)

    qmoe_lang.MTPDecoderLayer = MTPDecoderLayer
    qmoe_lang.MTPModule = MTPModule


def _patch_language_model(qlang: Any) -> None:
    cls = qlang.LanguageModel
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        n_mtp = _mtp_layer_count(args, config)
        if n_mtp > 0:
            try:
                from vmlx_engine.patches.mlx_lm_mtp import is_mtp_active

                active = is_mtp_active()
            except Exception:
                active = False
            if active:
                self.mtp = qlang.MTPModule(args)

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        n_confirmed: int = 0,
        **kwargs,
    ):
        force_text_rope = bool(getattr(self, "_vmlx_force_text_rope_1d", False))
        if not force_text_rope and not return_hidden and return_logits and n_confirmed == 0:
            return original_call(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                **kwargs,
            )

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        has_media_grid = image_grid_thw is not None or video_grid_thw is not None
        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None
            self._vmlx_text_rope_only = False

        cache_offset = 0
        if cache and cache[self.model.fa_idx] is not None:
            offset = cache[self.model.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, qlang.mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            if force_text_rope and not has_media_grid and (
                getattr(self, "_vmlx_text_rope_only", False)
                or self._rope_deltas is None
            ):
                batch_size, seq_length = inputs.shape
                if rope_mask is not None:
                    position_ids = qlang.mx.cumsum(
                        rope_mask.astype(qlang.mx.int64), axis=-1
                    ) - 1
                    position_ids = qlang.mx.where(
                        rope_mask == 0,
                        qlang.mx.ones_like(position_ids),
                        position_ids,
                    )
                else:
                    position_ids = qlang.mx.arange(seq_length).reshape(1, -1)
                    position_ids = qlang.mx.broadcast_to(
                        position_ids, (batch_size, seq_length)
                    )
                if cache_offset:
                    position_ids = position_ids + cache_offset
                self._rope_deltas = qlang.mx.zeros((batch_size,), dtype=qlang.mx.int64)
                self._vmlx_text_rope_only = True
            elif (
                (
                    cache is not None
                    and cache[self.model.fa_idx] is not None
                    and (cache_offset == 0)
                )
                or self._rope_deltas is None
                or cache is None
            ):
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
                    self._vmlx_text_rope_only = False
            else:
                batch_size, seq_length = inputs.shape
                delta = qlang.mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = qlang.mx.arange(seq_length).reshape(1, -1)
                position_ids = qlang.mx.broadcast_to(
                    position_ids, (batch_size, seq_length)
                )

                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = qlang.mx.expand_dims(delta, axis=0)
                    if delta.shape[0] < batch_size:
                        delta = qlang.mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                position_ids = qlang.mx.add(position_ids, delta)[None, ...]
                position_ids = qlang.mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )
                self._vmlx_text_rope_only = False

        hidden = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            n_confirmed=n_confirmed,
            return_unnormed=force_text_rope
            or return_hidden
            or not return_logits
            or n_confirmed > 0,
        )
        if not return_logits:
            return hidden
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        if return_hidden:
            return logits, hidden
        return qlang.LanguageModelOutput(logits=logits)

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(mtp_out)
        else:
            logits = self.lm_head(mtp_out)
        if return_hidden:
            return logits, mtp_out
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def sanitize(self, weights):
        if not hasattr(self, "mtp"):
            weights = {
                key: value
                for key, value in weights.items()
                if ".mtp." not in key and not key.startswith("mtp.")
            }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
            weights.pop("language_model.lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )

        sanitized = {}
        for key, value in weights.items():
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in norm_keys):
                if value.ndim == 1:
                    value = value + 1.0
            sanitized[key] = value
        return sanitized

    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        if (
            getattr(self.args, "num_experts", 0) <= 0
            and _mtp_layer_count(self.args, self.config) <= 0
        ):
            return None
        return predicate

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.quant_predicate = property(quant_predicate)
    cls._vmlx_mtp_patched = True


def _patch_moe_language_model(qmoe_lang: Any) -> None:
    cls = qmoe_lang.LanguageModel
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    original_init = cls.__init__

    def __init__(self, args, config=None):
        original_init(self, args, config)
        n_mtp = _mtp_layer_count(args, config)
        if n_mtp > 0:
            try:
                from vmlx_engine.patches.mlx_lm_mtp import is_mtp_active

                active = is_mtp_active()
            except Exception:
                active = False
            if active:
                self.mtp = qmoe_lang.MTPModule(args)

    cls.__init__ = __init__
    cls._vmlx_mtp_patched = True


def _patch_outer_model(qvl: Any) -> None:
    cls = qvl.Model
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    def __call__(
        self,
        input_ids,
        pixel_values=None,
        mask=None,
        cache=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        n_confirmed: int = 0,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        kwargs.update(
            {
                "pixel_values": pixel_values,
                **input_embeddings_features.to_dict(),
            }
        )
        return self.language_model(
            input_ids,
            mask=mask,
            cache=cache,
            return_hidden=return_hidden,
            return_logits=return_logits,
            n_confirmed=n_confirmed,
            **kwargs,
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache, return_hidden=return_hidden
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def sanitize(self, weights):
        norms_are_mlx_ready = bool(getattr(self, "_vmlx_norms_are_mlx_ready", False))
        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif key.startswith("mtp."):
                key = "language_model." + key
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            is_norm = any(key.endswith(sfx) for sfx in norm_keys)
            is_mtp_norm = ".mtp." in key and is_norm
            if (not norms_are_mlx_ready or is_mtp_norm) and is_norm:
                if value.ndim == 1:
                    value = value + 1.0
            sanitized_weights[key] = value

        if getattr(getattr(self.config, "text_config", None), "tie_word_embeddings", False):
            sanitized_weights.pop("language_model.lm_head.weight", None)
        return sanitized_weights

    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls._vmlx_mtp_patched = True


def _patch_moe_outer_model(qmoe_vl: Any) -> None:
    cls = qmoe_vl.Model
    if "_vmlx_mtp_patched" in cls.__dict__:
        return

    def __call__(
        self,
        input_ids,
        pixel_values=None,
        mask=None,
        cache=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        kwargs.update(
            {
                "pixel_values": pixel_values,
                **input_embeddings_features.to_dict(),
            }
        )
        return self.language_model(
            input_ids,
            mask=mask,
            cache=cache,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
            **kwargs,
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache, return_hidden=return_hidden
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def sanitize(self, weights):
        import mlx.core as mx

        weights = dict(weights)
        num_layers = int(getattr(self.config.text_config, "num_hidden_layers", 0) or 0)

        def split_fused(prefix: str) -> None:
            fused = f"{prefix}.experts.gate_up_proj"
            if fused in weights:
                gate_weight, up_weight = mx.split(weights.pop(fused), 2, axis=-2)
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weight
            down = f"{prefix}.experts.down_proj"
            if down in weights:
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(down)

        for layer_idx in range(num_layers):
            split_fused(f"model.language_model.layers.{layer_idx}.mlp")
            split_fused(f"language_model.model.layers.{layer_idx}.mlp")

        for key in list(weights.keys()):
            marker = ".mlp.experts.gate_up_proj"
            if not key.startswith("mtp.layers.") or not key.endswith(marker):
                continue
            prefix = key[: -len(".experts.gate_up_proj")]
            gate_weight, up_weight = mx.split(weights.pop(key), 2, axis=-2)
            weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
            weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weight
            down = f"{prefix}.experts.down_proj"
            if down in weights:
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(down)

        norms_are_mlx_ready = bool(getattr(self, "_vmlx_norms_are_mlx_ready", False))
        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif key.startswith("mtp."):
                key = "language_model." + key
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            is_norm = any(key.endswith(sfx) for sfx in norm_keys)
            is_mtp_norm = ".mtp." in key and is_norm
            if (not norms_are_mlx_ready or is_mtp_norm) and is_norm:
                if value.ndim == 1:
                    value = value + 1.0
            sanitized_weights[key] = value

        if getattr(getattr(self.config, "text_config", None), "tie_word_embeddings", False):
            sanitized_weights.pop("language_model.lm_head.weight", None)
        return sanitized_weights

    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls._vmlx_mtp_patched = True

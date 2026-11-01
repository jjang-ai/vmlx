# SPDX-License-Identifier: Apache-2.0
"""Native-MTP hooks for mlx-lm's Qwen3.5 / 3.6 / 3.8 text models.

Installed on ``mlx_lm.models.qwen3_5`` (and the MoE ``qwen3_5_moe.Model``):

``TextModelArgs.from_dict``
    keeps ``mtp_num_hidden_layers`` (top level or under ``text_config``), which
    ``BaseModelArgs`` would otherwise drop as an unknown key.
``MTPDecoderLayer`` / ``MTPModule``
    the Qwen MTP head: RMS-normalised token embedding and backbone hidden,
    concatenated, projected by ``fc`` and run through full-attention layers.
``GatedDeltaNet``
    mlx-lm's linear-attention forward expressed as a per-chunk step, so a
    verify forward (``n_confirmed``) can stop after the confirmed row, keep that
    state for a rejection, and record how to rebuild any accepted prefix.
``DecoderLayer`` / ``Qwen3_5TextModel``
    thread ``n_confirmed`` and the GDN capture sink down; the backbone returns
    the hidden state before the final norm (the head consumes it).
``TextModel`` / ``Model``
    head attachment, ``return_hidden`` / ``return_logits``, ``mtp_forward``,
    ``make_mtp_cache``, MTP-aware ``sanitize`` and ``quant_predicate``.
``qwen3_5_moe.Model.sanitize``
    unfuses or stacks routed-expert tensors for backbone and head layers.

mlx-vlm's separate Qwen copy is handled by ``patches/mlx_vlm_mtp``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx

from vmlx_engine.metal.gdn_conv_decode import (
    fused_qwen35_gdn_conv_requested,
    qwen35_gdn_conv_decode,
)
from vmlx_engine.metal.qwen35_gdn_gate_terms import (
    fused_qwen35_gdn_gates_requested,
    qwen35_gated_delta_decode,
)
from vmlx_engine.native_mtp_prompt_priming import capture_prefill, capture_requested

from ._marks import is_marked, mark

logger = logging.getLogger(__name__)

_installed = False

_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
)


def install() -> bool:
    """Hook ``mlx_lm.models.qwen3_5`` for native MTP. Idempotent."""
    global _installed
    if _installed:
        return True
    try:
        from mlx_lm.models import qwen3_5
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5 not importable; Qwen MTP hooks skipped")
        return False

    if hasattr(qwen3_5.TextModel, "mtp_forward") and not is_marked(qwen3_5.TextModel):
        # This mlx-lm already ships the head interface.
        mark(qwen3_5.TextModel, "upstream")
        _installed = True
        return True

    _hook_args(qwen3_5)
    _define_head_classes(qwen3_5)
    _hook_linear_attention(qwen3_5)
    _hook_decoder_layer(qwen3_5)
    _hook_backbone(qwen3_5)
    _hook_text_model(qwen3_5)
    _hook_wrapper_model(qwen3_5)
    _hook_moe_sanitize()

    _installed = True
    mark(qwen3_5.TextModel, "patch")
    logger.info("Qwen3.5/3.6 MTP model adapter applied")
    return True


# -- args --------------------------------------------------------------------


def _hook_args(qwen3_5: Any) -> None:
    args_cls = qwen3_5.TextModelArgs
    if is_marked(args_cls):
        return
    base_from_dict = args_cls.from_dict.__func__

    def from_dict(cls, params):
        args = base_from_dict(cls, params)
        layers = params.get("mtp_num_hidden_layers")
        nested = params.get("text_config")
        if layers is None and isinstance(nested, dict):
            layers = nested.get("mtp_num_hidden_layers")
        args.mtp_num_hidden_layers = int(layers or 0)
        return args

    args_cls.from_dict = classmethod(from_dict)
    mark(args_cls)


# -- head --------------------------------------------------------------------


def _define_head_classes(qwen3_5: Any) -> None:
    if hasattr(qwen3_5, "MTPModule"):
        return
    import mlx.nn as nn

    Attention = qwen3_5.Attention
    SparseMoeBlock = qwen3_5.SparseMoeBlock
    MLP = qwen3_5.MLP
    create_attention_mask = qwen3_5.create_attention_mask

    class MTPDecoderLayer(nn.Module):
        """Full-attention decoder layer used inside the head (never GDN)."""

        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = (
                SparseMoeBlock(args)
                if args.num_experts > 0
                else MLP(args.hidden_size, args.intermediate_size)
            )

        def __call__(self, x, mask=None, cache=None):
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Qwen next-token head: predicts t+2 from hidden(t) and token t+1."""

        def __init__(self, args):
            super().__init__()
            dim, eps = args.hidden_size, args.rms_norm_eps
            self.pre_fc_norm_hidden = nn.RMSNorm(dim, eps=eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(dim, eps=eps)
            self.fc = nn.Linear(dim * 2, dim, bias=False)
            self.layers = [MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)]
            self.norm = nn.RMSNorm(dim, eps=eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            token_part = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
            hidden_part = self.pre_fc_norm_hidden(hidden_states)
            x = self.fc(mx.concatenate([token_part, hidden_part], axis=-1))
            if cache is None:
                cache = [None] * len(self.layers)
            mask = create_attention_mask(x, cache[0] if cache else None)
            for layer, layer_cache in zip(self.layers, cache):
                x = layer(x, mask, layer_cache)
            return self.norm(x)

    qwen3_5.MTPDecoderLayer = MTPDecoderLayer
    qwen3_5.MTPModule = MTPModule


# -- GatedDeltaNet -----------------------------------------------------------


def _hook_linear_attention(qwen3_5: Any) -> None:
    cls = qwen3_5.GatedDeltaNet
    if is_marked(cls):
        return
    import mlx.nn as nn
    from mlx.nn.layers.distributed import sum_gradients
    from mlx_lm.models.gated_delta import gated_delta_update

    from .hybrid_state import arm_partial_rollback, remember_partial_rollback

    def conv_window(self, qkv, conv_state):
        """Unfused conv of ``qkv`` after ``conv_state``; returns (conv_in, conv_out)."""
        conv_in = mx.concatenate([conv_state, qkv], axis=1)
        return conv_in, nn.silu(self.conv1d(conv_in))

    def split_heads(self, conv_out, rows):
        B = conv_out.shape[0]
        q, k, v = [
            t.reshape(B, rows, heads, dim)
            for t, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k, v

    def recurrent_step(self, qkv, a, b, conv_state, ssm_state, ssm_mask=None, lengths=None):
        """Conv + gated delta update over ``qkv`` rows from the given states.

        Returns ``(out, conv_state, ssm_state)`` after the rows.
        """
        rows = qkv.shape[1]
        fused = None
        use_fused_conv = getattr(self, "_fused_gdn_conv", None)
        if use_fused_conv is None:
            use_fused_conv = fused_qwen35_gdn_conv_requested()
        if lengths is None and use_fused_conv and not self.training:
            fused = qwen35_gdn_conv_decode(qkv, conv_state, self.conv1d.weight, enabled=True)
        if fused is not None:
            conv_out, next_conv = fused
        else:
            conv_in = mx.concatenate([conv_state, qkv], axis=1)
            keep = self.conv_kernel_size - 1
            if lengths is not None:
                ends = mx.clip(lengths, 0, rows)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                next_conv = mx.take_along_axis(conv_in, positions, axis=1)
            else:
                next_conv = mx.contiguous(conv_in[:, -keep:])
            conv_out = nn.silu(self.conv1d(conv_in))

        q, k, v = split_heads(self, conv_out, rows)

        use_fused_gates = getattr(self, "_fused_gdn_gates", None)
        if use_fused_gates is None:
            use_fused_gates = fused_qwen35_gdn_gates_requested()
        update = None
        if use_fused_gates and not self.training:
            update = qwen35_gated_delta_decode(
                q, k, v, a, b, self.A_log, self.dt_bias, ssm_state, ssm_mask, enabled=True
            )
        if update is None:
            update = gated_delta_update(
                q, k, v, a, b, self.A_log, self.dt_bias, ssm_state, ssm_mask,
                use_kernel=not self.training,
            )
        out, next_ssm = update
        return out, next_conv, next_ssm

    def __call__(
        self,
        inputs: Any,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
        n_confirmed: int = 0,
    ):
        B, S, _ = inputs.shape
        if cache is not None:
            arm_partial_rollback(cache)
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        ssm_state = cache[1] if cache else None
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        def rows_of(x, lo, hi=None):
            if x is None or (lo == 0 and hi is None):
                return x
            return x[:, lo:hi]

        def sink(lo, hi, conv_in_state, ssm_in_state):
            # Hand the unfused per-chunk inputs to a prefill capture consumer.
            if gdn_sink is None:
                return
            chunk = rows_of(qkv, lo, hi)
            conv_in, conv_out = conv_window(self, chunk, conv_in_state)
            q, k, v = split_heads(self, conv_out, chunk.shape[1])
            gdn_sink.append((
                q, k, v, rows_of(a, lo, hi), rows_of(b, lo, hi), self.A_log, self.dt_bias,
                ssm_in_state, rows_of(mask, lo, hi), conv_in, self.conv_kernel_size,
            ))

        n = n_confirmed
        if 0 < n < S:
            sink(0, n, conv_state, ssm_state)
            out_c, conv_c, ssm_c = self.recurrent_step(
                qkv[:, :n], a[:, :n], b[:, :n], conv_state, ssm_state, rows_of(mask, 0, n)
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)

                def rebuild(accepted):
                    # Re-run only this layer's recurrent step over the accepted
                    # draft rows, starting from the confirmed snapshot.
                    if accepted == 0:
                        return conv_c, ssm_c
                    end = n + accepted
                    _, conv_r, ssm_r = self.recurrent_step(
                        qkv[:, n:end], a[:, n:end], b[:, n:end], conv_c, ssm_c,
                        rows_of(mask, n, end),
                    )
                    return conv_r, ssm_r

                remember_partial_rollback(cache, S - n, rebuild)
            sink(n, None, conv_c, ssm_c)
            out_d, conv_last, ssm_last = self.recurrent_step(
                qkv[:, n:], a[:, n:], b[:, n:], conv_c, ssm_c, rows_of(mask, n)
            )
            out = mx.concatenate([out_c, out_d], axis=1)
        else:
            lengths = cache.lengths if cache is not None else None
            sink(0, None, conv_state, ssm_state)
            out, conv_last, ssm_last = self.recurrent_step(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_last
            cache[1] = ssm_last
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out

    cls.recurrent_step = recurrent_step
    cls.__call__ = __call__
    mark(cls)


# -- decoder layer / backbone ------------------------------------------------


def _hook_decoder_layer(qwen3_5: Any) -> None:
    cls = qwen3_5.DecoderLayer
    if is_marked(cls):
        return

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        gdn_sink: Optional[list] = None,
        n_confirmed: int = 0,
    ):
        normed = self.input_layernorm(x)
        if self.is_linear:
            r = self.linear_attn(normed, mask, cache, gdn_sink=gdn_sink, n_confirmed=n_confirmed)
        else:
            try:
                r = self.self_attn(normed, mask, cache, position_ids)
            except TypeError:
                r = self.self_attn(normed, mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    cls.__call__ = __call__
    mark(cls)


def _hook_backbone(qwen3_5: Any) -> None:
    cls = qwen3_5.Qwen3_5TextModel
    if is_marked(cls):
        return
    create_attention_mask = qwen3_5.create_attention_mask
    create_ssm_mask = qwen3_5.create_ssm_mask

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        gdn_sink: Optional[list] = None,
        n_confirmed: int = 0,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = create_attention_mask(h, cache[self.fa_idx])
        recurrent_mask = create_ssm_mask(h, cache[self.ssm_idx])
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(
                h,
                mask=recurrent_mask if layer.is_linear else attention_mask,
                cache=layer_cache,
                gdn_sink=gdn_sink,
                n_confirmed=n_confirmed,
            )
        # Pre-norm: TextModel applies model.norm before the vocabulary head.
        return h

    cls.__call__ = __call__
    mark(cls)


# -- TextModel / Model -------------------------------------------------------


def _norms_look_raw(weights: dict) -> bool:
    """Whether a shard's backbone norms still use the HF (unshifted) convention."""
    for key, value in weights.items():
        if not key.endswith(_NORM_SUFFIXES) or getattr(value, "ndim", 0) != 1:
            continue
        try:
            head = value[: min(int(value.shape[0]), 1024)].astype(mx.float32)
            mean = float(mx.mean(head).item())
        except Exception:
            continue
        if key.endswith("model.norm.weight") and mean < 1.5:
            return True
        if mean < 0.5:
            return True
    return False


def _is_head_key(key: str) -> bool:
    return key.startswith("mtp.") or ".mtp." in key


def _hook_text_model(qwen3_5: Any) -> None:
    cls = qwen3_5.TextModel
    if is_marked(cls):
        return
    from mlx_lm.models.cache import KVCache

    from . import draft_head_loading_enabled

    base_init = cls.__init__

    def __init__(self, args):
        base_init(self, args)
        # Without the head the model is a stock build: sanitize drops mtp.*,
        # and the generation lane sees no head and stays out of the way.
        if int(getattr(args, "mtp_num_hidden_layers", 0) or 0) > 0 and draft_head_loading_enabled():
            self.mtp = qwen3_5.MTPModule(args)

    def vocab_logits(self, h):
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        gdn_sink: Optional[list] = None,
        n_confirmed: int = 0,
    ):
        hidden = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            gdn_sink=gdn_sink,
            n_confirmed=n_confirmed,
        )
        # Prompt priming folds an ordinary token prefill into the head history.
        # Seed/verify forwards (return_hidden) and media embeddings never do.
        if not return_hidden and input_embeddings is None and capture_requested(self):
            capture_prefill(self, inputs, hidden, cache)
        if not return_logits:
            return hidden
        logits = vocab_logits(self, self.model.norm(hidden))
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
        head_hidden = self.mtp(hidden_states, next_token_ids, self.model.embed_tokens, mtp_cache)
        logits = vocab_logits(self, head_hidden)
        if return_hidden:
            return logits, head_hidden
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def sanitize(self, weights):
        # The JANG loader sanitizes one shard at a time: a shard without mtp.*
        # keys says nothing about the bundle (native_mtp validates that up
        # front). Head norms ship in the raw HF convention even when the
        # backbone norms in the same shard are already MLX-shifted, so the
        # shift decision for the backbone ignores head tensors.
        if not hasattr(self, "mtp"):
            weights = {k: v for k, v in weights.items() if "mtp." not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        backbone = {k: v for k, v in weights.items() if not _is_head_key(k)}
        raw_conv = any(
            "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
            for k, v in backbone.items()
        )
        shift_backbone_norms = raw_conv or _norms_look_raw(backbone)
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
            if not key.endswith(_NORM_SUFFIXES) or value.ndim != 1:
                continue
            if _is_head_key(key) or shift_backbone_norms:
                weights[key] = value + 1.0
        return weights

    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            # A quantized mtp.fc must bind to a quantized Linear.
            return True

        has_head = int(getattr(self.args, "mtp_num_hidden_layers", 0) or 0) > 0
        if self.args.num_experts <= 0 and not has_head:
            return None
        return predicate

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls.quant_predicate = property(quant_predicate)
    mark(cls)


def _hook_wrapper_model(qwen3_5: Any) -> None:
    """VLM-shaped ``Model`` forwards to its hooked ``language_model``."""
    cls = qwen3_5.Model
    if is_marked(cls):
        return

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        gdn_sink: Optional[list] = None,
        n_confirmed: int = 0,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            gdn_sink=gdn_sink,
            n_confirmed=n_confirmed,
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache, return_hidden=return_hidden
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    mark(cls)


# -- qwen3_5_moe -------------------------------------------------------------


def _unfuse_experts(weights: dict, prefix: str) -> None:
    fused = weights.pop(f"{prefix}.experts.gate_up_proj", None)
    if fused is None:
        return
    half = fused.shape[-2] // 2
    weights[f"{prefix}.switch_mlp.gate_proj.weight"] = fused[..., :half, :]
    weights[f"{prefix}.switch_mlp.up_proj.weight"] = fused[..., half:, :]
    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(f"{prefix}.experts.down_proj")


def _stack_experts(weights: dict, prefix: str, count: int) -> None:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(
            [weights.pop(f"{prefix}.experts.{e}.{proj}.weight") for e in range(count)]
        )


def _hook_moe_sanitize() -> None:
    try:
        from mlx_lm.models import qwen3_5_moe
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5_moe not importable; dense Qwen only")
        return
    cls = qwen3_5_moe.Model
    if is_marked(cls):
        return

    def sanitize(self, weights):
        renamed = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            renamed[key] = value

        args = self.language_model.args
        for layer in range(args.num_hidden_layers):
            _unfuse_experts(renamed, f"language_model.model.layers.{layer}.mlp")

        head_layers = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        if head_layers > 0:
            # Qwen3.6 ships fused head experts, Qwen3.5 per-expert tensors.
            fused = "language_model.mtp.layers.0.mlp.experts.gate_up_proj" in renamed
            for layer in range(head_layers):
                prefix = f"language_model.mtp.layers.{layer}.mlp"
                if f"{prefix}.switch_mlp.gate_proj.weight" in renamed:
                    continue  # converter already stored switch_mlp layout
                if fused:
                    _unfuse_experts(renamed, prefix)
                else:
                    _stack_experts(renamed, prefix, args.num_experts)

        return self.language_model.sanitize(renamed)

    cls.sanitize = sanitize
    mark(cls)

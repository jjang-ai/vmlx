# Copyright © 2023-2024 Apple Inc.
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Derived from mlx-lm 0.31.3, ``mlx_lm/models/ernie4_5_moe.py`` (MIT License,
# https://github.com/ml-explore/mlx-lm/blob/main/LICENSE). The Apple copyright
# notice above and the MIT permission notice below apply to the portions
# retained from that file; the vMLX modifications are Apache-2.0.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""ERNIE-4.5 MoE text runtime for vMLX (``model_type: ernie4_5_moe``).

Vendored from mlx-lm 0.31.3 ``mlx_lm/models/ernie4_5_moe.py`` (the file shipped in that
release) and installed over the upstream
module by ``register.py``. Kept byte-for-byte where possible so
future upstream diffs are easy to read. Deliberate departures, each with the evidence that
motivated it (measurements in the PR that introduced this runtime):

1. Router bias is KEPT. Upstream drops ``mlp.moe_statics.e_score_correction_bias`` in
   ``sanitize`` and picks top-k on raw softmax probabilities. transformers (the reference that
   produced our parity logits), vLLM and Baidu FastDeploy all pick top-k on
   ``probs + bias`` and weight the chosen experts by the UNBIASED probs (DeepSeek-V3 style
   aux-loss-free load balancing: the bias steers selection only). Measured on
   ERNIE-4.5-21B-A3B-PT over 8 reference sequences: without the bias 29.0% of
   (token, MoE-layer) pairs choose a different expert set and final top-1 agreement with the
   reference falls to 0.955-0.989 on 7 of 8 prompts (max |delta logit| 6.05). The bias is F32,
   per layer nearly constant (+22..+63) with a within-layer spread of 0.005-0.03, which is
   exactly the scale of near-tied softmax probs, so it cannot be dropped as "almost constant".
2. Router logits are computed in float32 from the unquantized gate, matching transformers'
   ``F.linear(hidden.float(), weight.float())``. bf16 router logits have ~3 significant
   digits, and near-ties at the top-k boundary sit below that resolution (same reason as 1).
   When the gate is quantized we fall back to the module call.
3. ``moe_norm_min`` is read from the config (1e-12 here) instead of hard-coded.
4. ``make_cache`` is explicit: plain GQA, one ``KVCache`` per layer.
5. Native MTP head (``num_nextn_predict_layers`` = 1) is attached as ``Model.mtp`` and
   exposed through vMLX's duck-typed contract (``mtp_forward``, ``make_mtp_cache``, and
   ``__call__(return_hidden=True)`` returning the PRE-norm hidden like the Qwen3.5 patch).
   Checkpoint keys ``model.mtp_block.0.*`` / ``model.mtp_emb_norm.0`` /
   ``model.mtp_hidden_norm.0`` / ``model.mtp_linear_proj.0`` are renamed in ``sanitize`` to
   ``mtp.layers.0.*`` / ``mtp.emb_norm`` / ``mtp.hidden_norm`` / ``mtp.linear_proj`` so JANG
   tier rules written for ``mtp.layers`` apply unchanged. Head conventions (Baidu FastDeploy
   layout, chosen by an 8-way ablation against transformers, 2026-09-04): backbone hidden is
   taken AFTER ``model.norm`` (so ``mtp_forward`` applies it to the pre-norm hidden it is
   handed); fused = ``linear_proj(concat[emb_norm(embed(t+1)), hidden_norm(h_t)])``,
   embedding first; position-0 embedding kept; one dense decoder block (attention + 12,288
   SwiGLU, interleaved RoPE via its own KVCache offset); the block output goes through the
   SHARED ``model.norm`` before the tied lm_head. vLLM differs (zeroes pos-0, no output norm)
   and measured worse. Draft depth defaults to 1 (Baidu asserts one step). For depth 2/3
   chained drafting the engine feeds ``mtp_forward``'s returned hidden back in as the next
   step's "previous hidden", so that hidden is the head block's PRE-norm output (the shared
   norm is applied only on the way to the logits), exactly as for the backbone.
6. Weight loading has an explicit completion boundary. ``sanitize`` only renames
   tensors and buffers incomplete expert groups across shards. ``load_weights(strict=True)``
   or ``finalize_ernie_weight_loading`` checks that every parameter was supplied before
   accepting a model. Only then may a completely absent MTP head become AR-only. Missing
   routing biases require re-conversion; legacy raw-softmax routing is an explicit opt-in
   with ``VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS=1``. Partial or quantised biases fail.

Architecture (from config.json of ERNIE-4.5-21B-A3B-PT): 28 layers, hidden 2560, layer 0 dense
SwiGLU (12,288), layers 1-27 MoE with 64 routed experts (top-6, 1536 wide) + 2 shared experts,
GQA 20/4, interleaved RoPE theta 500k, tied embeddings, vocab 103,424.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

logger = logging.getLogger(__name__)


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    model_type: str
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    use_bias: bool
    tie_word_embeddings: bool
    moe_num_experts: int
    moe_layer_start_index: int = 0
    moe_intermediate_size: int = 0
    moe_capacity: list[int] = field(default_factory=list)
    moe_k: int = 1
    moe_layer_interval: int = 1
    moe_use_aux_free: bool = False
    moe_num_shared_experts: int = 0
    moe_layer_end_index: Optional[int] = None
    head_dim: Optional[int] = None
    moe_gate_act: str = "softmax"
    moe_norm_min: float = 1e-12  # vMLX: from config (departure 3)
    num_nextn_predict_layers: int = 0  # vMLX: MTP head count (departure 5)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.use_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.use_bias)

        # traditional=True is MLX's name for interleaved (pairwise) rotation, which is what
        # transformers' Ernie4_5 rotate_half does (x[0::2], x[1::2]).
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Ernie4_5_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Ernie4_5_MoeMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.k = args.moe_k
        self.norm_min = args.moe_norm_min
        self.moe_intermediate_size = (
            args.moe_intermediate_size
            if args.moe_intermediate_size
            else args.intermediate_size
        )

        self.gate = nn.Linear(args.hidden_size, args.moe_num_experts, bias=False)
        # vMLX departure 1: selection bias, F32, filled from
        # mlp.moe_statics.e_score_correction_bias by sanitize(). Not an nn.Linear bias.
        self.e_score_correction_bias = mx.zeros((args.moe_num_experts,), dtype=mx.float32)

        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            self.moe_intermediate_size,
            args.moe_num_experts,
            bias=args.use_bias,
        )

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                args.moe_intermediate_size * args.moe_num_shared_experts
                if getattr(args, "moe_intermediate_size", None)
                else args.intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_MLP(
                args.hidden_size, shared_intermediate_size, args.use_bias
            )
        else:
            self.shared_experts = None

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{args.moe_gate_act} is not supported.")

    def _router_logits(self, x: mx.array) -> mx.array:
        # vMLX departure 2: float32 router matmul when the gate is unquantized.
        w = getattr(self.gate, "weight", None)
        if w is not None and not hasattr(self.gate, "scales"):
            return mx.matmul(x.astype(mx.float32), w.astype(mx.float32).T)
        return self.gate(x).astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        probs = self.gate_act(self._router_logits(x))  # [B, L, E] float32

        k = self.k
        # vMLX departure 1: choose on probs + bias, weight by unbiased probs.
        choice = probs + self.e_score_correction_bias.astype(mx.float32)
        inds = mx.stop_gradient(mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(probs, inds, axis=-1)

        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), self.norm_min)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(y.dtype)).sum(axis=-2).astype(y.dtype)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)

        moe_layer_start_index = (
            min(args.moe_layer_start_index)
            if isinstance(args.moe_layer_start_index, (tuple, list))
            else args.moe_layer_start_index
        )

        if args.moe_layer_end_index is None:
            moe_layer_end_index = args.num_hidden_layers - 1
        else:
            moe_layer_end_index = (
                max(args.moe_layer_end_index)
                if isinstance(args.moe_layer_end_index, (tuple, list))
                else args.moe_layer_end_index
            )

        if (
            ((layer_idx + 1) % args.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeMLP(args)
        else:
            self.mlp = Ernie4_5_MLP(
                args.hidden_size, args.intermediate_size, args.use_bias
            )

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Ernie4_5_MTP(nn.Module):
    """One-step MTP head (departure 5). ``layers[0]`` is a dense decoder block: passing
    ``layer_idx=0`` to Ernie4_5_DecoderLayer selects the dense MLP because layer 0 sits below
    ``moe_layer_start_index``; the head's 12,288-wide MLP matches ``intermediate_size``."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.emb_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.linear_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.layers = [Ernie4_5_DecoderLayer(args, layer_idx=0) for _ in range(args.num_nextn_predict_layers)]

    def __call__(self, pre_norm_hidden, next_token_ids, embed_tokens, final_norm, cache=None):
        e = self.emb_norm(embed_tokens(next_token_ids))
        h = self.hidden_norm(final_norm(pre_norm_hidden))
        fused = self.linear_proj(mx.concatenate([e, h], axis=-1))
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(fused, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache):
            fused = layer(fused, mask, c)
        # PRE-norm head output: the caller applies the shared final norm for the
        # logits and hands this raw state to the next chained draft step.
        return fused


class Ernie45Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Ernie4_5_DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        # vMLX departure 5: return the PRE-norm residual stream; Model applies
        # self.norm for logits and hands the pre-norm tensor to the MTP head.
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self._ernie_loaded_keys = set()
        object.__setattr__(self, "_ernie_pending_experts", {})
        self.model_type = args.model_type
        self.model = Ernie45Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.num_nextn_predict_layers > 0:
            from vmlx_engine.native_mtp import native_mtp_disabled_by_env

            # Text dispatch uses head presence as its runtime eligibility gate.
            # Honor both disable aliases before sanitize/load, as Qwen does.
            if not native_mtp_disabled_by_env():
                self.mtp = Ernie4_5_MTP(args)

    def _logits(self, normed: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        **_ignored,
    ):
        hidden = self.model(inputs, cache)
        # Native-MTP prompt priming hook (same placement as the Qwen3.5 patch):
        # only a normal prompt prefill is folded, never seed/verify forwards.
        if not return_hidden and hasattr(self, "mtp"):
            try:
                from vmlx_engine.native_mtp_prompt_priming import capture_prefill, capture_requested
                if capture_requested(self):
                    capture_prefill(self, inputs, hidden, cache)
            except Exception:
                pass
        if not return_logits:
            return hidden
        out = self._logits(self.model.norm(hidden))
        if return_hidden:
            return out, hidden
        return out

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache, return_hidden=False):
        """vMLX native-MTP contract. ``hidden_states`` is a PRE-norm hidden: the backbone's
        (from ``__call__(return_hidden=True)``) or, for chained drafting, the raw head output
        this method returned on the previous step. The head normalises it; the returned
        hidden is likewise the head block's PRE-norm output (the shared final norm is applied
        only on the way to the logits), so feeding it back in does not norm twice."""
        mtp_out = self.mtp(hidden_states, next_token_ids, self.model.embed_tokens, self.model.norm, mtp_cache)
        logits = self._logits(self.model.norm(mtp_out))
        if return_hidden:
            return logits, mtp_out
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # vMLX departure 4: plain GQA, one KV cache per layer, no hybrid state.
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights):
        mtp_patterns = ("mtp_block.", "mtp_linear_proj.", "mtp_hidden_norm.", "mtp_emb_norm.")
        has_head = hasattr(self, "mtp")

        def _rename_mtp(key: str) -> str:
            # model.mtp_block.0.X -> mtp.layers.0.X ; model.mtp_emb_norm.0.weight -> mtp.emb_norm.weight
            k = key[len("model."):] if key.startswith("model.") else key
            if k.startswith("mtp_block."):
                return "mtp.layers." + k[len("mtp_block."):]
            for name in ("mtp_emb_norm", "mtp_hidden_norm", "mtp_linear_proj"):
                if k.startswith(name + "."):
                    rest = k[len(name) + 1:].split(".", 1)[1]  # drop the ".0" index
                    return "mtp." + name[len("mtp_"):] + "." + rest
            return key

        out = {}
        for key, value in weights.items():
            if not has_head and key.startswith("mtp."):
                continue
            if any(pattern in key for pattern in mtp_patterns):
                if has_head:
                    out[_rename_mtp(key)] = value
                continue
            if "e_score_correction_bias" in key and (
                key.endswith(".scales") or key.endswith(".biases") or key.endswith(".weight")
            ):
                # A quantised routing bias is a converter defect (jang_tools convert.py
                # float32 passthrough rule). Loading it silently left the runtime bias
                # at zeros and routing without it; fail loudly instead.
                raise ValueError(
                    f"{key}: e_score_correction_bias must be a float32 passthrough tensor, "
                    "not a quantised weight/scales/biases triplet; re-convert with the "
                    "fixed jang_tools (routing-fp32 passthrough)"
                )
            if key.endswith(".mlp.moe_statics.e_score_correction_bias"):
                # vMLX departure 1: keep, as [E] float32 on the MoE block.
                new_key = key.replace(".moe_statics.e_score_correction_bias", ".e_score_correction_bias")
                out[new_key] = value.reshape(-1).astype(mx.float32)
                continue
            out[key] = value
        weights = out

        if self.args.tie_word_embeddings:
            for suffix in ("weight", "scales", "biases"):
                weights.pop(f"lm_head.{suffix}", None)

        # A JANG load can split an expert group across files, including its
        # quantization sidecars. Retain those tensors until a complete stack
        # can be emitted under a real module path; never pass experts.N keys
        # to nn.Module.load_weights(strict=False), which would discard them.
        pending = self._ernie_pending_experts
        n_exp = self.args.moe_num_experts
        for l in range(self.args.num_hidden_layers):
            for m in ("gate_proj", "down_proj", "up_proj"):
                base = f"model.layers.{l}.mlp"
                for suffix in ("weight", "scales", "biases"):
                    target = f"{base}.switch_mlp.{m}.{suffix}"
                    keys = [f"{base}.experts.{e}.{m}.{suffix}" for e in range(n_exp)]
                    present = {k: weights.pop(k) for k in keys if k in weights}
                    if present:
                        pending.setdefault(target, {}).update(present)
                    group = pending.get(target)
                    if group is not None and target in weights:
                        raise ValueError(f"ERNIE checkpoint mixes stacked and individual experts: {target}")
                    if group is not None and all(k in group for k in keys):
                        weights[target] = mx.stack([group[k] for k in keys])
                        del pending[target]

        return weights

    def _prepare_complete_weight_load(self, keys):
        """Validate a complete load, never an individual shard. Return legacy defaults."""
        if self._ernie_pending_experts:
            groups = ", ".join(sorted(self._ernie_pending_experts))
            raise ValueError(f"ERNIE checkpoint has incomplete expert groups: {groups}")
        expected = set(dict(tree_flatten(self.parameters())))
        head = {k for k in expected if k.startswith("mtp.")}
        biases = {k for k in expected if k.endswith(".mlp.e_score_correction_bias")}
        missing = expected - keys
        required_missing = missing - head - biases
        if required_missing:
            raise ValueError("ERNIE checkpoint missing parameters: " + ", ".join(sorted(required_missing)))
        if head & keys and head - keys:
            raise ValueError("ERNIE checkpoint has an incomplete MTP head: " + ", ".join(sorted(head - keys)))
        defaults = {}
        if missing & biases:
            allow_legacy = os.environ.get("VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS", "").strip() == "1"
            if biases & keys or not allow_legacy:
                raise ValueError(
                    "ERNIE checkpoint missing routing bias (e_score_correction_bias). "
                    "Re-convert from the original HF checkpoint to preserve expert selection. "
                    "For legacy upstream conversions with ALL routing biases absent, explicitly "
                    "set VMLX_ERNIE45_ALLOW_MISSING_ROUTING_BIAS=1 to accept raw-softmax routing."
                )
            logger.warning(
                "ERNIE legacy compatibility enabled: all routing biases are absent; using "
                "raw-softmax routing. This changed 29%% of expert choices in the measured "
                "%s checkpoint. Re-convert from HF to restore the biases.",
                "ERNIE-4.5-21B-A3B-PT",
            )
            defaults = {k: mx.zeros((self.args.moe_num_experts,), dtype=mx.float32) for k in biases}
        if head and not head & keys:
            logger.warning("ERNIE checkpoint has no MTP head tensors; loading AR-only")
            delattr(self, "mtp")
        return defaults

    def load_weights(self, file_or_weights, strict=True):
        weights = dict(mx.load(file_or_weights) if isinstance(file_or_weights, str) else file_or_weights)
        if strict:
            weights.update(self._prepare_complete_weight_load(set(weights)))
        # Preserve MLX's strict shape validation for whole-model loads.
        result = super().load_weights(list(weights.items()), strict=strict)
        expected = set(dict(tree_flatten(self.parameters())))
        self._ernie_loaded_keys.update(expected & weights.keys())
        return result

    def finalize_ernie_weight_loading(self):
        """Called by vMLX after ALL JANG shards, including strict=False loads."""
        defaults = self._prepare_complete_weight_load(self._ernie_loaded_keys)
        if defaults:
            super().load_weights(list(defaults.items()), strict=False)
            self._ernie_loaded_keys.update(defaults)

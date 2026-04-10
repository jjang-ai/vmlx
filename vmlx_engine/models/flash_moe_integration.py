# SPDX-License-Identifier: Apache-2.0
"""Flash MoE model integration — patches SwitchLinear for on-demand expert loading.

Wraps MoE layers to intercept expert computation and load expert weights
from SSD via FlashMoEExpertLoader instead of keeping all experts in RAM.

The key insight: SwitchLinear stores all expert weights as a 3D tensor
[num_experts, out_features, in_features]. Flash MoE replaces this with
on-demand loading — only the top-K experts needed for each token are
loaded from disk (or served from slot-bank cache).

Compatible with:
    - Qwen 3.5 (SwitchLinear, 3-projection MoE)
    - Mistral 4 (SwitchLinear, 3-projection MoE)
    - MiniMax M2.5 (SwitchLinear, 3-projection, 256 experts)
    - Gemma 4 (SwitchLinear, separate router)
    - Nemotron (SwitchLinear, 2-projection, latent MoE)

Does NOT interact with:
    - JANG loader (_pre_fix_bits_from_shard operates at load time, before this)
    - TurboQuant (KV cache compression, independent of weight loading)
    - SSM companion cache (token-keyed, not weight-keyed)
    - Prefix cache (stores KV tensors, not model weights)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from ..flash_moe_config import FlashMoEConfig
    from ..utils.flash_moe_loader import FlashMoEExpertLoader

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared per-expert matmul — used by both FlashMoEBlock and the Gemma 4 shim
# ═══════════════════════════════════════════════════════════════════════════════


def _infer_bits_from_shapes(
    proj_tensors: Dict,
    in_dim: int,
    default_gs: int,
    native_bits: Optional[int],
) -> int:
    """Compute bits from packed weight shape given known in_dim."""
    weight = proj_tensors["weight"]
    packed_cols = weight.shape[-1]
    if in_dim > 0:
        bits = (packed_cols * 32) // in_dim
        if bits in (2, 3, 4, 5, 6, 8):
            return bits
    scales = proj_tensors.get("scales")
    if scales is not None:
        s_cols = scales.shape[-1]
        for try_gs in (default_gs, 64, 128):
            in_d = s_cols * try_gs
            if in_d > 0 and (packed_cols * 32) % in_d == 0:
                bits = (packed_cols * 32) // in_d
                if bits in (2, 3, 4, 5, 6, 8):
                    return bits
    return native_bits or 4


def _apply_expert_tensors(
    x: mx.array,
    ews,
    default_group_size: int,
    native_bits: Optional[int],
    activation_fn,
    is_switchglu: bool,
) -> mx.array:
    """Apply one expert's projections to flat input tokens.

    Handles JANG mixed-precision (per-projection bit widths) and both
    SwitchGLU (gate+up+down) and SwitchMLP (up+down only) structures.
    """
    tensors = ews.tensors
    has_gate = "gate_proj" in tensors
    has_up = "up_proj" in tensors
    has_down = "down_proj" in tensors

    def _matmul(x_in: mx.array, proj_tensors: Dict, in_dim: int) -> mx.array:
        weight = proj_tensors["weight"]
        if "scales" in proj_tensors:
            scales = proj_tensors["scales"]
            biases = proj_tensors.get("biases")
            bits = _infer_bits_from_shapes(
                proj_tensors, in_dim, default_group_size, native_bits
            )
            return mx.quantized_matmul(
                x_in, weight, scales, biases,
                group_size=default_group_size, bits=bits, transpose=True,
            )
        return x_in @ weight.T

    hidden_size = x.shape[-1]

    if has_gate and has_up and has_down:
        gate_out = _matmul(x, tensors["gate_proj"], hidden_size)
        up_out = _matmul(x, tensors["up_proj"], hidden_size)
        if is_switchglu and activation_fn is not None:
            hidden = activation_fn(up_out, gate_out)
        elif activation_fn is not None:
            hidden = activation_fn(gate_out) * up_out
        else:
            hidden = nn.silu(gate_out) * up_out
        return _matmul(hidden, tensors["down_proj"], hidden.shape[-1])
    elif has_up and has_down:
        up_out = _matmul(x, tensors["up_proj"], hidden_size)
        if activation_fn is not None:
            hidden = activation_fn(up_out)
        else:
            hidden = nn.silu(up_out)
        return _matmul(hidden, tensors["down_proj"], hidden.shape[-1])
    else:
        logger.warning(
            "Flash MoE: incomplete expert projections for layer %d expert %d "
            "(has gate=%s up=%s down=%s)",
            ews.layer_idx, ews.expert_idx, has_gate, has_up, has_down,
        )
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# FlashMoEBlock — wraps an entire MoE block for on-demand expert loading
# ═══════════════════════════════════════════════════════════════════════════════


class FlashMoEBlock(nn.Module):
    """Wraps an MoE block to use Flash MoE expert loading.

    Keeps the original routing logic (gate, top-k selection, score computation)
    but replaces expert weight computation with on-demand SSD loading.

    Supports all routing styles: softmax, sigmoid, pre_routed.
    """

    def __init__(
        self,
        original: nn.Module,
        loader: "FlashMoEExpertLoader",
        layer_idx: int,
    ):
        super().__init__()
        self.original = original
        self._loader = loader
        self._layer_idx = layer_idx

        # Detect routing style
        from ..utils.smelt_loader import _detect_routing_style
        self._routing_style = _detect_routing_style(original)

        # Detect activation + structure from the original switch_mlp/switch_glu
        # SwitchGLU (Qwen/Mistral/Gemma): activation takes (x_up, x_gate), default SwiGLU
        # SwitchMLP (Nemotron): activation takes (x) single arg, Nemotron uses ReLU²
        self._activation_fn = None  # None = use default silu*x for GLU, silu for simple
        self._is_glu = True  # gate+up+down vs up+down
        self._is_switchglu = False  # True if activation uses 2-arg convention
        switch = getattr(original, "switch_mlp", None) or getattr(original, "switch_glu", None)
        if switch is not None:
            act = getattr(switch, "activation", None)
            if act is not None:
                self._activation_fn = act
            # GLU has gate_proj; simple MoE (Nemotron SwitchMLP) doesn't
            self._is_glu = hasattr(switch, "gate_proj")
            # SwitchGLU uses 2-arg activation (SwiGLU(x, gate))
            self._is_switchglu = type(switch).__name__ == "SwitchGLU"

        # Detect bits/group_size from native quantized projections
        self._native_bits = None
        self._native_group_size = None
        if switch is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
                proj = getattr(switch, proj_name, None)
                if proj is not None and hasattr(proj, "bits"):
                    self._native_bits = proj.bits
                    self._native_group_size = proj.group_size
                    break

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with on-demand expert loading.

        Uses original routing to determine which experts are needed,
        then loads only those experts from SSD (or slot-bank cache).
        """
        orig = self.original

        # Resolve top-k
        ne: int = getattr(
            orig,
            "num_experts_per_tok",
            getattr(orig, "top_k", getattr(orig, "num_activated_experts", 8)),
        )

        # ── Routing (same as TurboRouteWrapper but without cache_bias) ──
        if self._routing_style == "sigmoid":
            gates = orig.gate(x.astype(mx.float32))
            scores = mx.sigmoid(gates)
            ecb = getattr(orig, "e_score_correction_bias", None)
            if ecb is not None:
                scores = scores + ecb
            inds = mx.argpartition(-scores, kth=ne - 1, axis=-1)[..., :ne]
            scores = mx.take_along_axis(mx.sigmoid(gates), inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = scores.astype(x.dtype)

        elif self._routing_style == "pre_routed":
            inds, scores = orig.gate(x)

        else:
            # Softmax (default)
            gates = orig.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            inds = mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            rsf = getattr(orig, "routed_scaling_factor", 1.0)
            scores = scores * rsf
            pes = getattr(orig, "per_expert_scale", None)
            if pes is not None:
                scores = scores * pes[inds]

        # ── Load experts on-demand ──
        unique_experts = set(inds.reshape(-1).tolist())
        expert_data = self._loader.load_experts_parallel(
            self._layer_idx, list(unique_experts)
        )

        # ── Latent projections (Nemotron) ──
        x_expert = x
        fc1 = getattr(orig, "fc1_latent_proj", None)
        fc2 = getattr(orig, "fc2_latent_proj", None)
        if fc1 is not None:
            x_expert = fc1(x)

        # ── Expert computation ──
        # Flatten batch dims
        orig_shape = x_expert.shape
        batch_size = 1
        for s in orig_shape[:-1]:
            batch_size *= s
        x_flat = x_expert.reshape(batch_size, -1)
        inds_flat = inds.reshape(batch_size, ne)
        in_features = x_flat.shape[-1]

        # Determine output features from first available expert
        out_features = in_features  # fallback
        for eidx, ews in expert_data.items():
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                if proj_name in ews.tensors and "weight" in ews.tensors[proj_name]:
                    w = ews.tensors[proj_name]["weight"]
                    out_features = w.shape[0]  # [out, in] or packed shape
                    break
            break

        # Compute MoE output using loaded experts
        y = self._compute_moe_output(
            x_flat, inds_flat, scores.reshape(batch_size, ne),
            expert_data, ne
        )

        # Reshape back
        y = y.reshape(list(orig_shape[:-1]) + [y.shape[-1]])

        if fc2 is not None:
            y = fc2(y)

        # ── Shared expert ──
        shared = getattr(orig, "shared_expert", getattr(orig, "shared_experts", None))
        if shared is not None:
            seg = getattr(orig, "shared_expert_gate", None)
            if seg is not None:
                y = y + mx.sigmoid(seg(x)) * shared(x)
            else:
                y = y + shared(x)

        return y

    def _compute_moe_output(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array,
        expert_data: Dict,
        top_k: int,
    ) -> mx.array:
        """Compute MoE output from on-demand loaded experts.

        For generation (batch=1): simple per-slot loop, minimal overhead.
        For prefill (batch>1): groups tokens by expert for batched matmuls.
        """
        batch_size = x.shape[0]
        indices_list = indices.tolist()

        # Fast path: batch=1 (token generation) — no grouping needed
        if batch_size == 1:
            row = indices_list[0]
            result = None
            for k in range(top_k):
                eidx = int(row[k]) if isinstance(row, list) else int(row)
                ews = expert_data.get(eidx)
                if ews is None:
                    continue
                expert_out = self._apply_expert(x, ews)
                weighted = expert_out * scores[0, k]
                result = weighted if result is None else result + weighted
            return result if result is not None else mx.zeros_like(x)

        # Batch path: group tokens per (expert, slot) for batched computation
        # Build per-slot expert groups to avoid scatter-add
        slot_outputs = []
        for k in range(top_k):
            # All tokens' expert index for this slot
            slot_expert_ids = [
                int(indices_list[b][k]) if isinstance(indices_list[b], list)
                else int(indices_list[b])
                for b in range(batch_size)
            ]
            slot_scores = scores[:, k: k + 1]  # [batch, 1]

            # Group tokens by expert within this slot
            expert_to_tokens: Dict[int, list] = {}
            for b, eidx in enumerate(slot_expert_ids):
                if eidx not in expert_to_tokens:
                    expert_to_tokens[eidx] = []
                expert_to_tokens[eidx].append(b)

            # Compute per-expert, then assemble slot output
            slot_out = mx.zeros_like(x)
            for eidx, token_idxs in expert_to_tokens.items():
                ews = expert_data.get(eidx)
                if ews is None:
                    continue
                idx_arr = mx.array(token_idxs)
                x_batch = x[idx_arr]  # [n, hidden]
                expert_out = self._apply_expert(x_batch, ews)  # [n, hidden]
                # Scatter back — for each token index, add expert output
                slot_out = slot_out.at[idx_arr].add(expert_out)

            slot_outputs.append(slot_out * slot_scores)

        # Sum across slots
        output = slot_outputs[0]
        for s in slot_outputs[1:]:
            output = output + s
        return output

    def _apply_expert(
        self, x: mx.array, ews
    ) -> mx.array:
        """Delegate to shared per-expert matmul routine."""
        return _apply_expert_tensors(
            x,
            ews,
            default_group_size=self._native_group_size or 128,
            native_bits=self._native_bits,
            activation_fn=self._activation_fn,
            is_switchglu=self._is_switchglu,
        )

    def _get_bits(self, ews) -> int:
        """Infer bits from the expert's weight tensor shape."""
        for proj_tensors in ews.tensors.values():
            weight = proj_tensors.get("weight")
            scales = proj_tensors.get("scales")
            if weight is not None and scales is not None and weight.dtype == mx.uint32:
                w_cols = weight.shape[-1]
                s_cols = scales.shape[-1]
                gs = self._get_group_size(ews)
                in_dim = s_cols * gs
                if in_dim > 0 and (w_cols * 32) % in_dim == 0:
                    bits = (w_cols * 32) // in_dim
                    if bits in (2, 3, 4, 5, 6, 8):
                        return bits
        return 4  # default fallback

    def _get_group_size(self, ews) -> int:
        """Infer group_size from the original model's config.

        Checks multiple attribute paths to handle all MoE families:
          - Qwen/Mistral/Gemma: switch_mlp.{gate_proj,up_proj,down_proj}
          - Nemotron: switch_mlp.{fc1,fc2} (renamed from up/down_proj)
          - Direct attributes on the MoE block itself
        """
        # Try switch_mlp sub-module first (most common)
        switch_mlp = getattr(self.original, "switch_mlp", None)
        if switch_mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
                proj = getattr(switch_mlp, proj_name, None)
                if proj is not None and hasattr(proj, "group_size"):
                    return proj.group_size

        # Try switch_glu (Gemma 4 variant)
        switch_glu = getattr(self.original, "switch_glu", None)
        if switch_glu is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(switch_glu, proj_name, None)
                if proj is not None and hasattr(proj, "group_size"):
                    return proj.group_size

        # Try direct group_size on the MoE block
        if hasattr(self.original, "group_size"):
            return self.original.group_size

        return 64  # default fallback


# ═══════════════════════════════════════════════════════════════════════════════
# FlashMoESwitchGLUShim — drop-in replacement for SwitchGLU that streams experts
#
# Used for Gemma 4: the DecoderLayer has a sibling Router + Experts(SwitchGLU),
# so we replace layer.experts.switch_glu in-place rather than wrapping the whole
# MoE block. Signature matches mlx_lm.models.switch_layers.SwitchGLU.__call__.
# ═══════════════════════════════════════════════════════════════════════════════


class FlashMoESwitchGLUShim(nn.Module):
    """Streams expert weights from SSD on-demand, matching SwitchGLU's signature.

    SwitchGLU expects (x, indices) where indices has shape [..., top_k] and
    returns [..., top_k, hidden]. The Gemma 4 Experts wrapper then multiplies
    by top_k_weights and sums across the top_k axis.

    This shim preserves that contract while loading experts per-token.
    """

    def __init__(
        self,
        original_switch_glu: nn.Module,
        loader: "FlashMoEExpertLoader",
        layer_idx: int,
    ):
        super().__init__()
        self._loader = loader
        self._layer_idx = layer_idx

        # Inherit activation from the original SwitchGLU (GeGLU for Gemma 4).
        self._activation_fn = getattr(original_switch_glu, "activation", None)

        # Detect bits/group_size from the original projections so per-expert
        # matmuls use the right kernel.
        self._native_bits = None
        self._native_group_size = None
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(original_switch_glu, proj_name, None)
            if proj is not None and hasattr(proj, "bits"):
                self._native_bits = proj.bits
                self._native_group_size = proj.group_size
                break

        # Free the original expert weights immediately — after the shim is
        # installed we only stream experts from SSD, so the 3-per-projection
        # SwitchLinear tensors are dead weight in RAM. This is the Gemma 4
        # equivalent of free_expert_weights().
        freed = 0
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(original_switch_glu, proj_name, None)
            if proj is None:
                continue
            for tensor_name in ("weight", "scales", "biases"):
                tensor = getattr(proj, tensor_name, None)
                if tensor is not None:
                    freed += tensor.nbytes
                    setattr(proj, tensor_name, mx.zeros((1,), dtype=mx.uint8))
        self._freed_bytes = freed

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """SwitchGLU contract: x=[..., H], indices=[..., K] → [..., K, H].

        Works for both prefill (ndim=3: [B, L, H]) and token-gen (ndim=2).
        """
        # Flatten all leading dims so we have x_flat=[T, H], idx_flat=[T, K]
        orig_shape = x.shape
        H = orig_shape[-1]
        K = indices.shape[-1]
        x_flat = x.reshape(-1, H)
        idx_flat = indices.reshape(-1, K)
        T = x_flat.shape[0]

        # Collect unique expert IDs and load in parallel (checks cache first)
        idx_list = idx_flat.tolist()
        unique: set = set()
        for row in idx_list:
            for e in (row if isinstance(row, list) else [row]):
                unique.add(int(e))
        expert_data = self._loader.load_experts_parallel(
            self._layer_idx, list(unique)
        )

        # Compute output per top-k slot. For each slot, group tokens by expert
        # ID so every expert runs exactly one batched matmul.
        slot_results: List[mx.array] = []
        for k in range(K):
            slot_out = mx.zeros_like(x_flat)
            expert_to_tokens: Dict[int, list] = {}
            for t, row in enumerate(idx_list):
                eidx = int(row[k]) if isinstance(row, list) else int(row)
                expert_to_tokens.setdefault(eidx, []).append(t)

            for eidx, toks in expert_to_tokens.items():
                ews = expert_data.get(eidx)
                if ews is None:
                    continue
                idx_arr = mx.array(toks)
                x_sub = x_flat[idx_arr]
                y_sub = _apply_expert_tensors(
                    x_sub,
                    ews,
                    default_group_size=self._native_group_size or 128,
                    native_bits=self._native_bits,
                    activation_fn=self._activation_fn,
                    is_switchglu=True,
                )
                slot_out = slot_out.at[idx_arr].add(y_sub)

            slot_results.append(slot_out)

        # Stack across K → [T, K, H], then unflatten leading dims
        stacked = mx.stack(slot_results, axis=1)
        return stacked.reshape(list(orig_shape[:-1]) + [K, H])


# ═══════════════════════════════════════════════════════════════════════════════
# Model patching — apply Flash MoE to all MoE layers
# ═══════════════════════════════════════════════════════════════════════════════


def apply_flash_moe(
    model: nn.Module,
    loader: "FlashMoEExpertLoader",
) -> int:
    """Patch all MoE layers in model to use Flash MoE expert loading.

    Replaces each MoE block with a FlashMoEBlock that loads experts
    on-demand from SSD instead of keeping them in RAM.

    After patching, expert weights can be freed from RAM — only the
    backbone (non-expert) weights remain resident.

    Args:
        model: The MLX model to patch.
        loader: FlashMoEExpertLoader with expert index and cache.

    Returns:
        Number of MoE layers patched.
    """
    from ..utils.smelt_loader import _find_moe_block, _get_layers_list

    try:
        layers = _get_layers_list(model)
    except ValueError:
        logger.warning("Flash MoE: could not find model layers")
        return 0

    patched = 0
    gemma4_patched = 0
    for layer_idx, layer in enumerate(layers):
        # Check if this layer has MoE
        if layer_idx not in loader._index.layers:
            continue

        # Gemma 4 path: router + experts are siblings on the DecoderLayer.
        # Replace layer.experts.switch_glu with the streaming shim so the
        # sibling Router keeps handling routing.
        if _is_gemma4_moe_layer(layer):
            experts = getattr(layer, "experts", None)
            original_glu = getattr(experts, "switch_glu", None) if experts is not None else None
            if original_glu is not None:
                shim = FlashMoESwitchGLUShim(
                    original_switch_glu=original_glu,
                    loader=loader,
                    layer_idx=layer_idx,
                )
                experts.switch_glu = shim
                gemma4_patched += 1
                patched += 1
                continue
            logger.warning(
                "Flash MoE: Gemma 4 layer %d has no experts.switch_glu — skipping",
                layer_idx,
            )
            continue

        moe_block, attr_name = _find_moe_block(layer)
        if moe_block is None:
            continue

        # Create Flash MoE wrapper
        flash_block = FlashMoEBlock(
            original=moe_block,
            loader=loader,
            layer_idx=layer_idx,
        )

        # Replace the MoE block in the layer
        if attr_name is not None:
            setattr(layer, attr_name, flash_block)
        patched += 1

    if gemma4_patched:
        logger.info(
            "Flash MoE: patched %d MoE layers (%d Gemma 4 router+experts, "
            "%d standard blocks)",
            patched, gemma4_patched, patched - gemma4_patched,
        )
    else:
        logger.info("Flash MoE: patched %d MoE layers", patched)
    return patched


def _is_gemma4_moe_layer(layer: nn.Module) -> bool:
    """Gemma 4 DecoderLayer with enable_moe=True exposes router + experts
    as direct siblings, with experts wrapping a SwitchGLU."""
    router = getattr(layer, "router", None)
    experts = getattr(layer, "experts", None)
    if router is None or experts is None:
        return False
    return hasattr(experts, "switch_glu")


def free_expert_weights(model: nn.Module) -> int:
    """Free expert weight tensors from RAM after Flash MoE patching.

    After apply_flash_moe(), expert weights are loaded on-demand from SSD.
    The original SwitchLinear weight tensors are still in RAM — this function
    replaces them with empty placeholders to free memory.

    Args:
        model: The MLX model (already patched with Flash MoE).

    Returns:
        Approximate bytes freed.
    """
    import gc

    from ..utils.smelt_loader import _get_layers_list

    try:
        layers = _get_layers_list(model)
    except ValueError:
        return 0

    freed_bytes = 0
    # Check known MoE attribute names — dir() on nn.Module misses dynamically set attrs
    _MOE_ATTR_NAMES = ("block_sparse_moe", "mlp", "mixer")

    def _zero_projections(switch_mod: nn.Module) -> int:
        local_freed = 0
        for proj_name in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
            proj = getattr(switch_mod, proj_name, None)
            if proj is None:
                continue
            for tensor_name in ("weight", "scales", "biases"):
                tensor = getattr(proj, tensor_name, None)
                if tensor is not None:
                    local_freed += tensor.nbytes
                    setattr(proj, tensor_name, mx.zeros((1,), dtype=mx.uint8))
        return local_freed

    for layer in layers:
        if not isinstance(layer, nn.Module):
            continue

        # Standard path: Qwen/Mistral/MiniMax/Nemotron wrap the MoE block
        # under block_sparse_moe / mlp / mixer.
        for attr_name in _MOE_ATTR_NAMES:
            attr = getattr(layer, attr_name, None)
            if not isinstance(attr, FlashMoEBlock):
                continue
            orig = attr.original
            switch = (
                getattr(orig, "switch_mlp", None)
                or getattr(orig, "switch_glu", None)
            )
            if switch is not None:
                freed_bytes += _zero_projections(switch)

        # Gemma 4 path: experts.switch_glu was replaced with FlashMoESwitchGLUShim,
        # which already zeroed the original projections in its __init__ and
        # recorded the byte count in shim._freed_bytes.
        experts_attr = getattr(layer, "experts", None)
        if experts_attr is not None:
            sg = getattr(experts_attr, "switch_glu", None)
            if isinstance(sg, FlashMoESwitchGLUShim):
                freed_bytes += getattr(sg, "_freed_bytes", 0)

    gc.collect()
    logger.info("Flash MoE: freed ~%.2f GB expert weights", freed_bytes / 1e9)
    return freed_bytes

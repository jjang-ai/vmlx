"""smelt_loader.py — Smelt mode components for partial expert loading.

TurboRouteWrapper wraps an existing MoE block and injects cache-bias routing
so that the model prefers loaded experts.  Expert compute is delegated to the
NATIVE compiled SwitchGLU/SwitchMLP kernel — no custom matmul, baseline speed.

Three routing styles:
  softmax   — Qwen 3.5, Mistral 4, Gemma 4 (default)
  sigmoid   — MiniMax M2.5, GLM-5 (e_score_correction_bias + normalize)
  pre_routed — Nemotron Cascade/Super (gate returns (indices, scores) directly;
               cache_bias NOT injected here — Nemotron's compiled gate owns it)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Supported MoE class names (used by _detect_routing_style / _find_moe_block)
# ---------------------------------------------------------------------------
_MOE_CLASSES = {
    "Qwen3NextSparseMoeBlock",
    "MiniMaxMoE",
    "NemotronHMoEBlock",
    "LatentNemotronHMoE",
    "Mistral4MoE",
    "GlmMoeDsaMoE",
    "Gemma4SparseMoeBlock",
}


# ---------------------------------------------------------------------------
# Helper: detect routing style from class name
# ---------------------------------------------------------------------------

def _detect_routing_style(moe_block: nn.Module) -> str:
    """Return the routing style for *moe_block* based on its class name.

    Returns one of "softmax", "sigmoid", or "pre_routed".
    """
    class_name = type(moe_block).__name__
    if "MiniMax" in class_name or "Glm" in class_name:
        return "sigmoid"
    if "NemotronH" in class_name or "DeepSeek" in class_name:
        return "pre_routed"
    return "softmax"


# ---------------------------------------------------------------------------
# Helper: find the MoE sub-block inside a transformer layer
# ---------------------------------------------------------------------------

def _find_moe_block(layer: nn.Module):
    """Return ``(block, attr_name)`` for the MoE sub-block inside *layer*.

    Tries the following attribute names in order:
      block_sparse_moe, mlp, mixer

    Also accepts a layer that *is itself* a SwitchMLP/SwitchGLU block
    (direct switch_mlp attribute).

    Returns ``(None, None)`` if no MoE block is found.
    """
    for attr in ("block_sparse_moe", "mlp", "mixer"):
        candidate = getattr(layer, attr, None)
        if candidate is None:
            continue
        class_name = type(candidate).__name__
        if class_name in _MOE_CLASSES:
            return candidate, attr
        # Fallback: anything that owns a switch_mlp is treated as MoE
        if hasattr(candidate, "switch_mlp"):
            return candidate, attr

    # Layer itself might be a direct MoE block (rare, but be defensive)
    if type(layer).__name__ in _MOE_CLASSES or hasattr(layer, "switch_mlp"):
        return layer, None

    return None, None


# ---------------------------------------------------------------------------
# TurboRouteWrapper
# ---------------------------------------------------------------------------

class TurboRouteWrapper(nn.Module):
    """Wraps an existing MoE block, injects cache_bias routing, remaps indices.

    Delegates expert compute to the NATIVE compiled SwitchGLU/SwitchMLP path
    so there is no speed penalty compared to a fully-loaded model.

    Args:
        original:       The original MoE block (e.g. Qwen3NextSparseMoeBlock).
        remap:          Integer array of shape ``(num_loaded_experts,)`` mapping
                        *global* expert index → *local slot* index.  Pass
                        ``None`` when all experts are loaded (identity mapping).
        cache_bias:     Float array of shape ``(num_experts,)`` where unloaded
                        experts receive ``-1000`` and loaded experts receive
                        ``0``.  Routing argpartition will strongly prefer loaded
                        experts.
        routing_style:  One of "softmax", "sigmoid", or "pre_routed".
    """

    def __init__(
        self,
        original: nn.Module,
        remap,
        cache_bias,
        routing_style: str = "softmax",
    ):
        super().__init__()
        self.original = original
        self.remap = remap
        self.cache_bias = cache_bias
        self.routing_style = routing_style

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        orig = self.original

        # Resolve top-k (different models use different attribute names)
        ne: int = getattr(
            orig,
            "num_experts_per_tok",
            getattr(orig, "top_k", getattr(orig, "num_activated_experts", 8)),
        )

        # ------------------------------------------------------------------
        # 1. Routing
        # ------------------------------------------------------------------
        if self.routing_style == "sigmoid":
            # MiniMax M2.5 / GLM-5 style
            gates = orig.gate(x.astype(mx.float32))
            ss = mx.sigmoid(gates)
            sel = ss
            ecb = getattr(orig, "e_score_correction_bias", None)
            if ecb is not None:
                sel = sel + ecb
            sel = sel + self.cache_bias
            inds = mx.argpartition(-sel, kth=ne - 1, axis=-1)[..., :ne]
            # Unbiased scores for weighting, then normalize
            scores = mx.take_along_axis(ss, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = scores.astype(x.dtype)

        elif self.routing_style == "pre_routed":
            # Nemotron Cascade / Super: gate returns (indices, scores) directly.
            # cache_bias is NOT injected — Nemotron's compiled gate owns routing.
            inds, scores = orig.gate(x)

        else:
            # Softmax — Qwen 3.5, Mistral 4, Gemma 4 (default)
            gates = orig.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            orig_gates = gates                              # unbiased
            gates = gates + self.cache_bias                 # biased for selection
            inds = mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne]
            # Use UNBIASED scores for weighting
            scores = mx.take_along_axis(orig_gates, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            rsf = getattr(orig, "routed_scaling_factor", 1.0)
            scores = scores * rsf

        # ------------------------------------------------------------------
        # 2. Remap global → local slot indices
        # ------------------------------------------------------------------
        local_inds = self.remap[inds] if self.remap is not None else inds

        # ------------------------------------------------------------------
        # 3. Optional latent projections (Nemotron latent MoE)
        # ------------------------------------------------------------------
        x_expert = x
        fc1 = getattr(orig, "fc1_latent_proj", None)
        fc2 = getattr(orig, "fc2_latent_proj", None)
        if fc1 is not None:
            x_expert = fc1(x)

        # ------------------------------------------------------------------
        # 4. NATIVE SwitchGLU / SwitchMLP forward — compiled, full speed
        # ------------------------------------------------------------------
        y = orig.switch_mlp(x_expert, local_inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if fc2 is not None:
            y = fc2(y)

        # ------------------------------------------------------------------
        # 5. Shared expert contribution
        # ------------------------------------------------------------------
        shared = getattr(
            orig, "shared_expert", getattr(orig, "shared_experts", None)
        )
        if shared is not None:
            seg = getattr(orig, "shared_expert_gate", None)
            if seg is not None:
                y = y + mx.sigmoid(seg(x)) * shared(x)
            else:
                y = y + shared(x)

        return y

# SPDX-License-Identifier: Apache-2.0
"""LatentMoE support for Nemotron-H models.

mlx-lm's nemotron_h.py does not implement latent MoE projections
(fc1_latent_proj / fc2_latent_proj) used by models like Nemotron-3-Super-120B.

These projections compress hidden states into a lower-dimensional latent space
before expert routing, then project back after expert computation.

Standard MoE:  input(4096) → experts(4096→2688→4096) → output
LatentMoE:     input(4096) → fc1_latent(4096→1024) → experts(1024→2688→1024) → fc2_latent(1024→4096) → output

This module monkey-patches nemotron_h to support LatentMoE when moe_latent_size
is present in the model config.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger("vmlx_engine")

_patched = False
_lock = threading.Lock()


def patch_nemotron_h_for_latent_moe():
    """
    Monkey-patch mlx-lm's nemotron_h module to support LatentMoE.

    Adds:
    - moe_latent_size field to ModelArgs
    - LatentMoE wrapper that adds fc1_latent_proj / fc2_latent_proj around SwitchMLP
    - Patched NemotronHBlock to use LatentMoE for expert blocks

    Safe to call repeatedly and forward-compatible: if mlx-lm adds native
    LatentMoE support (ModelArgs already has moe_latent_size), this is a no-op.
    """
    global _patched
    if _patched:
        return

    with _lock:
        if _patched:
            return

        try:
            import importlib
            from dataclasses import fields as dc_fields

            import mlx.nn as nn

            nemotron_h = importlib.import_module("mlx_lm.models.nemotron_h")
            switch_layers = importlib.import_module("mlx_lm.models.switch_layers")
            SwitchMLP = switch_layers.SwitchMLP

            # --- 0. Skip if mlx-lm already supports LatentMoE natively ---
            OrigModelArgs = nemotron_h.ModelArgs
            native_fields = {f.name for f in dc_fields(OrigModelArgs)}
            if "moe_latent_size" in native_fields:
                # Only set _patched if this is truly native support (not our
                # own PatchedModelArgs from a previous partial patch attempt).
                if not hasattr(OrigModelArgs, "_latent_moe_patched"):
                    _patched = True
                    logger.info(
                        "mlx-lm already supports moe_latent_size natively — "
                        "skipping vMLX LatentMoE patch"
                    )
                    return
                # If _latent_moe_patched is set, ModelArgs was patched by us
                # but block init may not have been. Fall through to patch it.

            # --- 1. Patch ModelArgs to accept moe_latent_size ---
            if not hasattr(OrigModelArgs, "_latent_moe_patched"):
                from dataclasses import dataclass
                from typing import ClassVar

                @dataclass
                class PatchedModelArgs(OrigModelArgs):
                    moe_latent_size: Optional[int] = None
                    _latent_moe_patched: ClassVar[bool] = True

                nemotron_h.ModelArgs = PatchedModelArgs
                logger.info("Patched nemotron_h.ModelArgs with moe_latent_size")

            # --- 2. Create LatentMoE NemotronHMoE replacement ---
            NemotronHMLP = nemotron_h.NemotronHMLP
            MoEGate = nemotron_h.MoEGate

            class LatentNemotronHMoE(nn.Module):
                """
                NemotronHMoE with latent projections.

                Forward:
                  1. Gate routes on full hidden_size (4096)
                  2. fc1_latent_proj compresses input (4096 → moe_latent_size)
                  3. SwitchMLP processes in latent space (1024 → 2688 → 1024)
                  4. fc2_latent_proj expands output (moe_latent_size → 4096)
                  5. Shared expert operates in full hidden_size (unchanged)
                """

                def __init__(self, args):
                    super().__init__()
                    self.config = args
                    self.num_experts_per_tok = args.num_experts_per_tok
                    latent_size = args.moe_latent_size

                    # Latent projections (shared across all experts)
                    self.fc1_latent_proj = nn.Linear(
                        args.hidden_size, latent_size, bias=False
                    )
                    self.fc2_latent_proj = nn.Linear(
                        latent_size, args.hidden_size, bias=False
                    )

                    # Experts operate in latent space
                    self.switch_mlp = SwitchMLP(
                        latent_size,
                        args.moe_intermediate_size,
                        args.n_routed_experts,
                        activation=nn.ReLU2(),
                    )

                    # Gate operates in full hidden_size
                    self.gate = MoEGate(args)

                    # Shared expert in full hidden_size (no latent compression)
                    if args.n_shared_experts is not None:
                        intermediate_size = args.moe_shared_expert_intermediate_size
                        self.shared_experts = NemotronHMLP(
                            args, intermediate_size=intermediate_size
                        )

                def __call__(self, x):
                    import mlx.core as mx

                    # Route in full dimension
                    inds, scores = self.gate(x)

                    # Compress to latent space
                    x_latent = self.fc1_latent_proj(x)

                    # Expert computation in latent space
                    y = self.switch_mlp(x_latent, inds)
                    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

                    # Expand back to full dimension
                    y = self.fc2_latent_proj(y)

                    # Shared expert in full dimension
                    if self.config.n_shared_experts is not None:
                        y = y + self.shared_experts(x)

                    return y

            # --- 3. Patch NemotronHBlock to use LatentMoE when applicable ---
            OrigNemotronHBlock = nemotron_h.NemotronHBlock

            # Guard against double-patching: if block init is already patched,
            # don't re-capture (would cause infinite recursion).
            if not getattr(OrigNemotronHBlock, "_latent_block_patched", False):
                _orig_block_init = OrigNemotronHBlock.__init__

                def _patched_block_init(self, args, block_type):
                    # Always call original init first — inherits norm, block_type,
                    # and any future additions to NemotronHBlock.__init__.
                    _orig_block_init(self, args, block_type)

                    if (
                        block_type == "E"
                        and getattr(args, "moe_latent_size", None) is not None
                    ):
                        # Replace standard MoE mixer with LatentMoE variant
                        self.mixer = LatentNemotronHMoE(args)

                OrigNemotronHBlock.__init__ = _patched_block_init
                OrigNemotronHBlock._latent_block_patched = True

            # Note: No sanitize() patch needed — latent proj weights
            # (fc1_latent_proj, fc2_latent_proj) load directly by name.
            # The original sanitize() handles conv1d transpose + expert stacking.

            _patched = True
            logger.info(
                "Patched nemotron_h for LatentMoE support "
                "(fc1_latent_proj / fc2_latent_proj)"
            )

        except Exception as e:
            logger.warning(f"Failed to patch nemotron_h for LatentMoE: {e}")
            raise


def needs_latent_moe_patch(model_path: str) -> bool:
    """Check if a model requires the LatentMoE patch."""
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Only nemotron_h has MoE blocks that need LatentMoE patching.
        # Plain "nemotron" is a dense LLaMA-like model with no MoE.
        return (
            config.get("model_type") == "nemotron_h"
            and config.get("moe_latent_size") is not None
        )
    except Exception:
        return False


def ensure_latent_moe_support(model_path: str) -> bool:
    """
    Check if model needs LatentMoE and apply patch if so.

    Returns True if patch was applied, False if not needed.
    """
    if needs_latent_moe_patch(model_path):
        patch_nemotron_h_for_latent_moe()
        return True
    return False

# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous ("puzzle") Nemotron-H blocks, e.g. Nemotron-3.5-Bolt-74B-A9B.

NVIDIA's nemotron_h_puzzle checkpoints give every MoE block its own ``n_routed_experts`` / ``moe_intermediate_size`` /
``num_experts_per_tok`` in ``config["block_configs"]`` (Bolt: 240-496 experts, 1536-2560 wide, top-9) and leave those
fields out of the global config. mlx-lm's nemotron_h builds every block from the one global ModelArgs, so it cannot
build these models. This patch (idempotent, only active when ``block_configs`` is present):

  * ModelArgs accepts ``block_configs``; missing global MoE fields are filled from the first MoE block so the stock
    constructor runs;
  * after the stock NemotronHModel constructor, every MoE block's mixer is rebuilt from its own merged args;
  * the router of those blocks computes its logits in float32, like the shipped HF code
    (``F.linear(hidden_states.float(), weight.float())``) — a bf16 logit can reorder near-tied experts.

Bundles carry the experts pre-stacked (``switch_mlp.fc1`` / ``fc2``), so mlx-lm's sanitize (which stacks
``experts.N`` with the global expert count) has nothing to do for them.

The float32 router is not puzzle-specific: NVIDIA's NemotronHTopkRouter casts both operands to float32 for every
Nemotron-H MoE (uniform ones too, e.g. Nemotron-3.5-Super-VL: 512 experts, top-22, where near-tied bf16 logits flip
expert choices). ``ensure_puzzle_support`` therefore installs the fp32 router for any nemotron_h config with routed
experts and the per-block rebuild only when ``block_configs`` is present.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Optional

logger = logging.getLogger("vmlx_engine")

_patched = False
_lock = threading.Lock()
BLOCK_FIELDS = ("n_routed_experts", "moe_intermediate_size", "num_experts_per_tok")


_router_patched = False


def is_nemotron_h_moe(config: dict) -> bool:
    if not isinstance(config, dict):
        return False
    c = config.get("llm_config") if isinstance(config.get("llm_config"), dict) else config
    return c.get("model_type") in ("nemotron_h", "nemotron_h_v2", "nemotron_h_puzzle") and bool(c.get("n_routed_experts") or has_block_configs(c))


def patch_nemotron_h_fp32_router() -> None:
    """Every nemotron_h MoEGate computes its logits in float32 (HF: F.linear(x.float(), weight.float()))."""
    global _router_patched
    if _router_patched:
        return
    with _lock:
        if _router_patched:
            return
        import importlib

        import mlx.core as mx

        nemotron_h = importlib.import_module("mlx_lm.models.nemotron_h")

        def fp32_call(self, x):
            return nemotron_h.group_expert_select(
                x.astype(mx.float32) @ self.weight.astype(mx.float32).T,
                self.e_score_correction_bias,
                self.top_k,
                self.n_group,
                self.topk_group,
                self.routed_scaling_factor,
                self.norm_topk_prob,
            )

        nemotron_h.MoEGate.__call__ = fp32_call
        _router_patched = True
        logger.info("Patched mlx_lm nemotron_h MoEGate: float32 router logits (HF reference)")


def has_block_configs(config: dict) -> bool:
    bc = config.get("block_configs") if isinstance(config, dict) else None
    return isinstance(bc, list) and any(isinstance(b, dict) and b.get("block_type") == "moe" for b in bc)


def patch_nemotron_h_for_puzzle() -> None:
    global _patched
    if _patched:
        return
    with _lock:
        if _patched:
            return
        import importlib
        from typing import ClassVar

        import mlx.core as mx

        nemotron_h = importlib.import_module("mlx_lm.models.nemotron_h")
        Orig = nemotron_h.ModelArgs
        if not hasattr(Orig, "_puzzle_patched"):
            orig_post = Orig.__post_init__

            @dataclasses.dataclass
            class PuzzleModelArgs(Orig):
                block_configs: Optional[list] = None
                _puzzle_patched: ClassVar[bool] = True

                def __post_init__(self):
                    if self.block_configs:
                        moe = [b for b in self.block_configs if b.get("block_type") == "moe"]
                        for f in BLOCK_FIELDS:
                            if getattr(self, f, None) is None and moe and f in moe[0]:
                                setattr(self, f, moe[0][f])
                        if self.layers_block_type is None and self.hybrid_override_pattern is None:
                            self.layers_block_type = [b["block_type"] for b in self.block_configs]
                    orig_post(self)

            nemotron_h.ModelArgs = PuzzleModelArgs

        class Float32Router(nemotron_h.MoEGate):
            def __call__(self, x):
                return nemotron_h.group_expert_select(
                    x.astype(mx.float32) @ self.weight.astype(mx.float32).T,
                    self.e_score_correction_bias,
                    self.top_k,
                    self.n_group,
                    self.topk_group,
                    self.routed_scaling_factor,
                    self.norm_topk_prob,
                )

        orig_init = nemotron_h.NemotronHModel.__init__

        def puzzle_init(self, args):
            orig_init(self, args)
            bcs = getattr(args, "block_configs", None)
            if not bcs:
                return
            if len(bcs) != len(self.layers):
                raise ValueError(f"nemotron_h puzzle: {len(bcs)} block_configs for {len(self.layers)} blocks")
            n = 0
            for i, (layer, bc) in enumerate(zip(self.layers, bcs)):
                if layer.block_type != "E":
                    continue
                if bc.get("block_type") != "moe":
                    raise ValueError(f"nemotron_h puzzle: block {i} is MoE in the pattern but {bc.get('block_type')} in block_configs")
                over = {f: bc[f] for f in BLOCK_FIELDS if f in bc}
                largs = dataclasses.replace(args, **over)
                layer.mixer = nemotron_h.NemotronHMoE(largs)
                layer.mixer.gate.__class__ = Float32Router
                n += 1
            logger.info("nemotron_h puzzle: %d MoE blocks built from their own block_configs (fp32 router)", n)

        nemotron_h.NemotronHModel.__init__ = puzzle_init
        _patched = True
        logger.info("Patched mlx_lm nemotron_h for per-block (puzzle) MoE configs")


def ensure_puzzle_support(config: dict) -> bool:
    """fp32 router for any nemotron_h MoE; per-block MoE rebuild for puzzle configs. True iff the puzzle patch applies."""
    if is_nemotron_h_moe(config):
        patch_nemotron_h_fp32_router()
    if not has_block_configs(config):
        return False
    patch_nemotron_h_for_puzzle()
    return True

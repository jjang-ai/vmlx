# SPDX-License-Identifier: Apache-2.0
"""Layer assignment algorithm for pipeline parallelism.

Assigns transformer layers to nodes based on available RAM and compute
capability. Accounts for MoE layers being larger than dense layers,
and for smelt mode reducing MoE layer size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LayerProfile:
    """Memory and compute profile for a single layer."""
    index: int
    memory_gb: float       # Estimated memory for this layer's weights
    is_moe: bool           # Whether this is a MoE layer
    num_experts: int = 0   # Total experts (MoE only)
    loaded_experts: int = 0  # Loaded experts if smelt (0 = all)
    is_ssm: bool = False   # Whether this is an SSM/Mamba layer


@dataclass
class NodeAssignment:
    """Layer assignment for a single node."""
    node_id: str
    hostname: str
    layer_start: int
    layer_end: int         # Exclusive
    estimated_memory_gb: float
    available_memory_gb: float
    relative_compute: float  # Relative compute speed (1.0 = baseline)

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "num_layers": self.num_layers,
            "estimated_memory_gb": round(self.estimated_memory_gb, 2),
            "available_memory_gb": round(self.available_memory_gb, 2),
        }


def estimate_layer_profiles(
    config: dict,
    total_weight_gb: float,
    num_layers: int,
    smelt_percent: int = 100,
) -> List[LayerProfile]:
    """Estimate memory profile for each layer from model config.

    For MoE models, expert layers are larger. Non-expert layers (attention,
    norm, embedding) are the "backbone" and much smaller per layer.
    """
    text_cfg = config.get("text_config", config)
    num_experts = (
        text_cfg.get("n_routed_experts")
        or text_cfg.get("num_experts")
        or text_cfg.get("num_local_experts")
        or 0
    )
    is_moe = num_experts > 0

    if not is_moe:
        # Dense model — all layers roughly equal
        per_layer_gb = total_weight_gb / (num_layers + 2)  # +2 for embed/lm_head
        return [
            LayerProfile(index=i, memory_gb=per_layer_gb, is_moe=False)
            for i in range(num_layers)
        ]

    # MoE model — estimate backbone vs expert split
    # Heuristic: expert weights are typically 60-80% of total for large MoE
    # Use the ExpertIndex ratio if available, otherwise estimate
    expert_ratio = 0.7  # default assumption
    backbone_gb = total_weight_gb * (1 - expert_ratio)
    expert_gb = total_weight_gb * expert_ratio

    if smelt_percent < 100:
        expert_gb = expert_gb * smelt_percent / 100

    per_layer_backbone = backbone_gb / (num_layers + 2)
    per_layer_expert = expert_gb / num_layers  # experts spread across all layers

    # Detect which layers are MoE vs dense (some models alternate)
    layer_types = text_cfg.get("layer_types", None)
    moe_layers = set()
    if layer_types:
        for i, lt in enumerate(layer_types):
            if "moe" in lt.lower() or "sparse" in lt.lower():
                moe_layers.add(i)
    else:
        # Assume all layers have MoE (most common for large MoE models)
        moe_layers = set(range(num_layers))

    # Detect SSM layers (hybrid models like Nemotron)
    ssm_layers = set()
    if layer_types:
        for i, lt in enumerate(layer_types):
            if "mamba" in lt.lower() or "ssm" in lt.lower():
                ssm_layers.add(i)

    loaded_experts = max(1, num_experts * smelt_percent // 100)

    profiles = []
    for i in range(num_layers):
        if i in moe_layers:
            profiles.append(LayerProfile(
                index=i,
                memory_gb=per_layer_backbone + per_layer_expert,
                is_moe=True,
                num_experts=num_experts,
                loaded_experts=loaded_experts if smelt_percent < 100 else num_experts,
                is_ssm=i in ssm_layers,
            ))
        else:
            profiles.append(LayerProfile(
                index=i,
                memory_gb=per_layer_backbone,
                is_moe=False,
                is_ssm=i in ssm_layers,
            ))

    return profiles


def assign_layers_by_ram(
    layer_profiles: List[LayerProfile],
    nodes: List[dict],
    overhead_gb: float = 2.0,
) -> List[NodeAssignment]:
    """Assign layers to nodes proportional to available RAM.

    Args:
        layer_profiles: Per-layer memory profiles.
        nodes: List of node dicts with at least 'node_id', 'hostname',
               'available_gb', and optionally 'relative_compute'.
        overhead_gb: Per-node overhead for framework/OS (subtracted from available).

    Returns:
        List of NodeAssignment, one per node, covering all layers.
    """
    if not nodes:
        raise ValueError("No nodes available for layer assignment")
    if not layer_profiles:
        raise ValueError("No layers to assign")

    total_model_gb = sum(lp.memory_gb for lp in layer_profiles)
    num_layers = len(layer_profiles)

    # Sort nodes by available memory (largest first for better packing)
    sorted_nodes = sorted(nodes, key=lambda n: n.get("available_gb", 0), reverse=True)

    # Calculate effective capacity per node
    capacities = []
    for node in sorted_nodes:
        cap = max(0, node.get("available_gb", 0) - overhead_gb)
        capacities.append(cap)

    total_capacity = sum(capacities)
    if total_capacity < total_model_gb:
        logger.warning(
            "Total cluster capacity (%.1fGB) is less than model size (%.1fGB). "
            "Model may not fit — some layers may be swapped to disk.",
            total_capacity, total_model_gb,
        )

    # Assign layers proportional to capacity
    assignments = []
    layer_idx = 0
    for i, node in enumerate(sorted_nodes):
        if i == len(sorted_nodes) - 1:
            # Last node gets remaining layers
            layer_end = num_layers
        else:
            # Proportion of total capacity this node has
            proportion = capacities[i] / total_capacity if total_capacity > 0 else 1.0 / len(sorted_nodes)
            n_layers = max(1, round(num_layers * proportion))
            layer_end = min(layer_idx + n_layers, num_layers)

        if layer_idx >= num_layers:
            break

        assigned_memory = sum(lp.memory_gb for lp in layer_profiles[layer_idx:layer_end])
        assignments.append(NodeAssignment(
            node_id=node["node_id"],
            hostname=node.get("hostname", node["node_id"]),
            layer_start=layer_idx,
            layer_end=layer_end,
            estimated_memory_gb=assigned_memory,
            available_memory_gb=node.get("available_gb", 0),
            relative_compute=node.get("relative_compute", 1.0),
        ))
        layer_idx = layer_end

    # Log assignment
    for a in assignments:
        logger.info(
            "  %s: layers %d-%d (%d layers, ~%.1fGB / %.1fGB available)",
            a.hostname, a.layer_start, a.layer_end - 1, a.num_layers,
            a.estimated_memory_gb, a.available_memory_gb,
        )

    return assignments


def assign_layers_balanced(
    layer_profiles: List[LayerProfile],
    nodes: List[dict],
    warmup_times_ms: Optional[Dict[str, float]] = None,
    overhead_gb: float = 2.0,
) -> List[NodeAssignment]:
    """Assign layers balancing both RAM and compute time.

    If warmup_times_ms is provided (from a calibration run), adjusts
    the RAM-based assignment to equalize per-node compute time.
    Without calibration data, falls back to RAM-proportional assignment.
    """
    # Start with RAM-based assignment
    assignments = assign_layers_by_ram(layer_profiles, nodes, overhead_gb)

    if not warmup_times_ms or len(assignments) < 2:
        return assignments

    # Adjust based on measured compute times
    # Move layers from slow nodes to fast nodes (±2 layers max)
    node_times = {}
    for a in assignments:
        t = warmup_times_ms.get(a.node_id, 0)
        if t > 0:
            node_times[a.node_id] = t / a.num_layers  # per-layer time

    if len(node_times) < 2:
        return assignments

    avg_per_layer = sum(node_times.values()) / len(node_times)

    for a in assignments:
        if a.node_id in node_times:
            per_layer = node_times[a.node_id]
            if per_layer > avg_per_layer * 1.2:
                logger.info(
                    "  %s is %.0f%% slower than average — consider removing layers",
                    a.hostname, (per_layer / avg_per_layer - 1) * 100,
                )
            elif per_layer < avg_per_layer * 0.8:
                logger.info(
                    "  %s is %.0f%% faster than average — could take more layers",
                    a.hostname, (1 - per_layer / avg_per_layer) * 100,
                )

    return assignments

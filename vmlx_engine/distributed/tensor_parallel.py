# SPDX-License-Identifier: Apache-2.0
"""Tensor parallelism using mlx.distributed.

MLX's distributed module provides GPU-to-GPU communication primitives
(all_sum, all_gather, send, recv) over a ring backend. This module
implements tensor-parallel model sharding where each rank holds a
slice of every layer's weights.

Transport:
- Local (same machine, multiple GPUs): shared memory ring
- Thunderbolt 5: network ring over TB5 bridge (RDMA-like latency)
- Ethernet/WiFi: network ring over TCP (higher latency)

The ring backend is initialized by launching N processes that connect
to each other. MLX handles the transport — we configure the topology.

Usage:
    # Launch on 2 machines via MPI-style launcher:
    vmlx-distributed --model path/to/model --tp 2

    # Or via SSH:
    # Machine 1: vmlx-distributed --model ... --rank 0 --world-size 2 --master-addr 192.168.1.10
    # Machine 2: vmlx-distributed --model ... --rank 1 --world-size 2 --master-addr 192.168.1.10
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def init_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> mx.distributed.Group:
    """Initialize MLX distributed backend.

    If rank/world_size are not provided, reads from environment:
    - VMLX_RANK or RANK or LOCAL_RANK
    - VMLX_WORLD_SIZE or WORLD_SIZE

    Returns the global communication group.
    """
    if rank is None:
        rank = int(os.environ.get("VMLX_RANK", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))))
    if world_size is None:
        world_size = int(os.environ.get("VMLX_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

    if world_size <= 1:
        logger.info("Single process — distributed disabled")
        return mx.distributed.init(strict=False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    group = mx.distributed.init(strict=True, backend="ring")
    logger.info(
        "Distributed initialized: rank %d/%d",
        group.rank(), group.size(),
    )
    return group


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-parallel weight sharding.

    Weight matrix W of shape (out, in) is split along the output dimension:
    each rank holds W[rank_start:rank_end, :]. The forward pass produces
    a partial output that needs no communication (each rank has independent
    output columns).

    Used for: q_proj, k_proj, v_proj, gate_proj, up_proj in attention and MLP.
    """

    def __init__(self, original: nn.Linear, group: mx.distributed.Group):
        super().__init__()
        self.group = group
        rank = group.rank()
        world_size = group.size()

        # Shard weight along output dimension
        out_features = original.weight.shape[0]
        shard_size = out_features // world_size
        start = rank * shard_size
        end = start + shard_size

        self.weight = original.weight[start:end]
        if original.bias is not None:
            self.bias = original.bias[start:end]
        else:
            self.bias = None

        # For quantized weights, shard similarly
        if hasattr(original, "scales"):
            self.scales = original.scales[start:end]
        if hasattr(original, "biases") and hasattr(original, "scales"):
            self.biases = original.biases[start:end]

    def __call__(self, x: mx.array) -> mx.array:
        # Standard matmul on sharded weight — no communication needed
        if hasattr(self, "scales"):
            # Quantized path
            return mx.quantized_matmul(
                x, self.weight, self.scales,
                getattr(self, "biases", None),
            )
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class RowParallelLinear(nn.Module):
    """Linear layer with row-parallel weight sharding.

    Weight matrix W of shape (out, in) is split along the input dimension:
    each rank holds W[:, rank_start:rank_end]. After the local matmul,
    an all_sum reduces partial results across ranks.

    Used for: o_proj, down_proj in attention and MLP.
    """

    def __init__(self, original: nn.Linear, group: mx.distributed.Group):
        super().__init__()
        self.group = group
        rank = group.rank()
        world_size = group.size()

        # Shard weight along input dimension (columns)
        in_features = original.weight.shape[1]
        shard_size = in_features // world_size
        start = rank * shard_size
        end = start + shard_size

        self.weight = original.weight[:, start:end]
        self.bias = original.bias if original.bias is not None else None

        if hasattr(original, "scales"):
            # Quantized weights: scales are per group_size along input dim
            # Need careful sharding to maintain group boundaries
            self.scales = original.scales
            if hasattr(original, "biases"):
                self.biases = original.biases

    def __call__(self, x: mx.array) -> mx.array:
        # Local matmul on sharded input
        y = x @ self.weight.T
        # AllReduce across ranks to combine partial results
        y = mx.distributed.all_sum(y, group=self.group)
        if self.bias is not None:
            y = y + self.bias
        return y


def shard_attention(attn_module, group: mx.distributed.Group):
    """Shard an attention module for tensor parallelism.

    Column-parallel: q_proj, k_proj, v_proj (independent heads per rank)
    Row-parallel: o_proj (needs all_sum to combine head outputs)
    """
    world_size = group.size()
    if world_size <= 1:
        return attn_module

    for name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn_module, name, None)
        if proj is not None and isinstance(proj, nn.Linear):
            setattr(attn_module, name, ColumnParallelLinear(proj, group))

    o_proj = getattr(attn_module, "o_proj", None)
    if o_proj is not None and isinstance(o_proj, nn.Linear):
        attn_module.o_proj = RowParallelLinear(o_proj, group)

    # Update head counts for this rank
    if hasattr(attn_module, "n_heads"):
        attn_module.n_heads = attn_module.n_heads // world_size
    if hasattr(attn_module, "n_kv_heads"):
        attn_module.n_kv_heads = max(1, attn_module.n_kv_heads // world_size)
    if hasattr(attn_module, "num_heads"):
        attn_module.num_heads = attn_module.num_heads // world_size
    if hasattr(attn_module, "num_key_value_heads"):
        attn_module.num_key_value_heads = max(1, attn_module.num_key_value_heads // world_size)

    return attn_module


def shard_mlp(mlp_module, group: mx.distributed.Group):
    """Shard an MLP module for tensor parallelism.

    Column-parallel: gate_proj, up_proj (independent slices per rank)
    Row-parallel: down_proj (needs all_sum to combine)
    """
    world_size = group.size()
    if world_size <= 1:
        return mlp_module

    for name in ["gate_proj", "up_proj", "gate", "up"]:
        proj = getattr(mlp_module, name, None)
        if proj is not None and isinstance(proj, nn.Linear):
            setattr(mlp_module, name, ColumnParallelLinear(proj, group))

    for name in ["down_proj", "down"]:
        proj = getattr(mlp_module, name, None)
        if proj is not None and isinstance(proj, nn.Linear):
            setattr(mlp_module, name, RowParallelLinear(proj, group))

    return mlp_module


def shard_model_tp(model, group: mx.distributed.Group):
    """Apply tensor parallelism to a full model.

    Shards attention and MLP in every transformer layer. Embedding
    and lm_head stay replicated (small relative to layer weights).
    """
    world_size = group.size()
    if world_size <= 1:
        return model

    # Find layers
    layers = None
    for accessor in [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.backbone.layers,
    ]:
        try:
            layers = accessor(model)
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError("Could not find model layers for TP sharding")

    sharded = 0
    for i, layer in enumerate(layers):
        # Shard attention
        attn = getattr(layer, "self_attn", getattr(layer, "attention", None))
        if attn is not None:
            shard_attention(attn, group)
            sharded += 1

        # Shard MLP (skip MoE — MoE has its own expert routing)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and not _is_moe_mlp(mlp):
            shard_mlp(mlp, group)

    logger.info(
        "Tensor parallelism: sharded %d layers across %d ranks",
        sharded, world_size,
    )
    return model


def _is_moe_mlp(mlp) -> bool:
    """Check if an MLP module is a MoE block (has expert routing)."""
    return (
        hasattr(mlp, "gate")
        or hasattr(mlp, "router")
        or hasattr(mlp, "switch_mlp")
        or hasattr(mlp, "switch_glu")
        or hasattr(mlp, "experts")
    )


def launch_distributed(
    model_path: str,
    world_size: int,
    hosts: List[str],
    port: int = 29500,
    cluster_secret: str = "",
) -> str:
    """Generate the launch command for multi-node TP.

    Returns a shell command string that the UI can display or execute.
    Uses MLX's built-in ring backend with SSH-based process launch.
    """
    if len(hosts) != world_size:
        raise ValueError(f"Need {world_size} hosts, got {len(hosts)}")

    # MLX ring backend uses environment variables for setup
    # Each process needs RANK and WORLD_SIZE
    commands = []
    for rank, host in enumerate(hosts):
        env = f"VMLX_RANK={rank} VMLX_WORLD_SIZE={world_size}"
        if cluster_secret:
            env += f" VMLX_CLUSTER_SECRET={cluster_secret}"
        cmd = f"{env} vmlx serve {model_path} --distributed --distributed-mode tensor"
        if host == "localhost" or host == "127.0.0.1":
            commands.append(cmd)
        else:
            commands.append(f"ssh {host} '{cmd}'")

    return " &\n".join(commands)

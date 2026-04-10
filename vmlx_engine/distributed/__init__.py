# SPDX-License-Identifier: Apache-2.0
"""Distributed inference across multiple Macs via Thunderbolt 5 / network.

Supports pipeline parallelism (layer splitting) and tensor parallelism
(weight sharding) with auto-discovery of connected nodes.
"""

from .engine import DistributedEngine
from .mesh_manager import MeshManager

__all__ = ["DistributedEngine", "MeshManager"]

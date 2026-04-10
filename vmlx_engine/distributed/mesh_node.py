# SPDX-License-Identifier: Apache-2.0
"""MeshNode — state machine for a single node in the vMLX mesh.

Each node in the mesh has a lifecycle: DISCOVERY → JOIN → READY →
ASSIGNED → LOADING → ACTIVE → IDLE. Heartbeats keep the mesh alive.
If the coordinator disappears, nodes trigger an election.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

from .discovery import NodeInfo, get_local_node_info, LinkType

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 5.0       # seconds between heartbeats
SUSPECT_TIMEOUT = 15.0         # seconds before marking node suspect
DEAD_TIMEOUT = 30.0            # seconds before marking node dead


class NodeState(Enum):
    DISCOVERY = "discovery"
    JOINING = "joining"
    READY = "ready"
    ASSIGNED = "assigned"
    LOADING = "loading"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPECT = "suspect"
    DEAD = "dead"
    SHUTDOWN = "shutdown"


class MeshNode:
    """Represents this node's state and handles heartbeat/lifecycle."""

    def __init__(
        self,
        cluster_secret: str = "",
        on_state_change: Optional[Callable] = None,
        on_coordinator_lost: Optional[Callable] = None,
    ):
        self.info = get_local_node_info()
        self.state = NodeState.DISCOVERY
        self.cluster_secret = cluster_secret
        self.coordinator_id: Optional[str] = None
        self.is_coordinator = False
        self.joined_at: Optional[float] = None
        self.last_heartbeat_sent: float = 0
        self.assigned_layers: Optional[tuple] = None
        self.model_path: Optional[str] = None

        self._on_state_change = on_state_change
        self._on_coordinator_lost = on_coordinator_lost
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def node_id(self) -> str:
        return self.info.node_id

    @property
    def capability_score(self) -> float:
        """Score used for coordinator election.

        Higher = more capable = more likely to be coordinator.
        Weighted: RAM (40%), GPU cores (30%), available GB (20%), uptime (10%).
        """
        uptime_hours = (time.time() - self.joined_at) / 3600 if self.joined_at else 0
        return (
            self.info.ram_gb * 0.4
            + self.info.gpu_cores * 0.3
            + self.info.available_gb * 0.2
            + min(uptime_hours, 24) * 0.1  # cap at 24h to prevent runaway
        )

    def set_state(self, new_state: NodeState):
        old = self.state
        self.state = new_state
        self.info.status = new_state.value
        logger.info("Node %s: %s → %s", self.info.hostname, old.value, new_state.value)
        if self._on_state_change:
            self._on_state_change(self, old, new_state)

    def become_coordinator(self):
        self.is_coordinator = True
        self.coordinator_id = self.node_id
        logger.info("Node %s elected as coordinator", self.info.hostname)

    def accept_coordinator(self, coordinator_id: str):
        self.is_coordinator = False
        self.coordinator_id = coordinator_id
        logger.info("Node %s accepted coordinator: %s", self.info.hostname, coordinator_id)

    async def start_heartbeat(self):
        """Start sending periodic heartbeats."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self):
        while True:
            self.last_heartbeat_sent = time.time()
            # Heartbeat is sent by MeshManager, not here
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    def to_announce(self) -> dict:
        """Build an ANNOUNCE message payload."""
        return {
            "node_id": self.node_id,
            "hostname": self.info.hostname,
            "address": self.info.address,
            "port": self.info.port,
            "chip": self.info.chip,
            "ram_gb": self.info.ram_gb,
            "gpu_cores": self.info.gpu_cores,
            "available_gb": self.info.available_gb,
            "vmlx_version": self.info.vmlx_version,
            "mlx_version": self.info.mlx_version,
            "state": self.state.value,
            "capability_score": self.capability_score,
            "is_coordinator": self.is_coordinator,
            "coordinator_id": self.coordinator_id,
            "assigned_layers": list(self.assigned_layers) if self.assigned_layers else None,
            "model_path": self.model_path,
        }

    def auth_token(self) -> str:
        """Generate an auth token from the cluster secret + node_id."""
        if not self.cluster_secret:
            return ""
        return hashlib.sha256(
            f"{self.cluster_secret}:{self.node_id}".encode()
        ).hexdigest()[:32]

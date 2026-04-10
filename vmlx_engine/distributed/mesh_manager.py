# SPDX-License-Identifier: Apache-2.0
"""MeshManager — top-level lifecycle manager for the vMLX compute mesh.

Ties together discovery, election, topology, and coordinator/worker roles
into a single cohesive run loop. This is the only class that server.py
needs to interact with.

Usage:
    manager = MeshManager(cluster_secret="...", model_path="/path/to/model")
    await manager.start()           # discover, elect, assign, load
    logits = await manager.forward(input_ids)  # distributed inference
    await manager.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from .coordinator import Coordinator
from .discovery import (
    BonjourAdvertiser,
    NodeInfo,
    discover_all,
    get_local_node_info,
    measure_bandwidth,
    detect_best_route,
)
from .layer_assign import (
    assign_layers_by_ram,
    assign_layers_balanced,
    estimate_layer_profiles,
)
from .mesh_election import MeshElection, ElectionResult
from .mesh_node import MeshNode, NodeState, HEARTBEAT_INTERVAL, DEAD_TIMEOUT
from .mesh_topology import MeshTopology, PeerInfo, LinkInfo
from .protocol import Message, MessageType
from .worker import Worker

logger = logging.getLogger(__name__)


class MeshManager:
    """Top-level lifecycle manager for distributed inference.

    Handles the full lifecycle:
    1. Start → discover peers → form mesh
    2. Elect coordinator (capability-scored)
    3. Assign layers → load model across nodes
    4. Run distributed inference (pipeline or tensor parallel)
    5. Handle node join/leave/failure
    """

    def __init__(
        self,
        cluster_secret: str = "",
        model_path: str = "",
        mode: str = "pipeline",       # "pipeline" or "tensor"
        smelt_percent: int = 100,
        worker_nodes: Optional[str] = None,  # "ip:port,ip:port" manual list
        on_topology_change: Optional[Callable] = None,
        on_status_change: Optional[Callable] = None,
        force_coordinator: bool = False,
    ):
        self.cluster_secret = cluster_secret
        self.model_path = model_path
        self.mode = mode
        self.smelt_percent = smelt_percent
        self.force_coordinator = force_coordinator
        self._on_topology_change = on_topology_change
        self._on_status_change = on_status_change

        # Parse manual worker nodes
        self._manual_nodes: List[Tuple[str, int]] = []
        if worker_nodes:
            for entry in worker_nodes.split(","):
                entry = entry.strip()
                if ":" in entry:
                    host, port = entry.rsplit(":", 1)
                    self._manual_nodes.append((host, int(port)))
                else:
                    self._manual_nodes.append((entry, 9100))

        # Core components
        self.node = MeshNode(cluster_secret=cluster_secret)
        self.topology = MeshTopology()
        self.coordinator: Optional[Coordinator] = None
        self.worker: Optional[Worker] = None

        # State
        self._is_coordinator = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._model_loaded = False
        self._model_config: Optional[dict] = None

    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._running

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "is_coordinator": self._is_coordinator,
            "model_loaded": self._model_loaded,
            "model_path": self.model_path,
            "mode": self.mode,
            "num_nodes": len(self.topology.active_nodes),
            "total_ram_gb": self.topology.total_ram_gb,
            "coordinator": self.topology.coordinator.hostname if self.topology.coordinator else None,
            "topology": self.topology.to_snapshot(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> bool:
        """Full startup: discover → elect → assign → load.

        Returns True if the mesh is ready for inference.
        """
        self._running = True
        self._emit_status("starting")

        # Step 1: Add ourselves to topology
        local = self.node.info
        self.topology.add_peer(PeerInfo(
            node_id=local.node_id,
            hostname=local.hostname,
            address="127.0.0.1",
            port=local.port,
            chip=local.chip,
            ram_gb=local.ram_gb,
            gpu_cores=local.gpu_cores,
            available_gb=local.available_gb,
            capability_score=self.node.capability_score,
        ))
        self.node.set_state(NodeState.DISCOVERY)

        # Step 2: Discover peers
        logger.info("Discovering peers...")
        self._emit_status("discovering")

        cached = [{"address": h, "port": p} for h, p in self._manual_nodes]
        discovered = await discover_all(
            timeout=5.0,
            cached_peers=cached if cached else None,
            on_found=lambda n, m: logger.info("  Found %s via %s (%s)", n.hostname, m, n.chip),
        )

        # Add discovered nodes to topology
        for node_info in discovered:
            # Measure link quality
            bw, lat = await measure_bandwidth(node_info.address, node_info.port)
            link_type = detect_best_route(node_info.address)

            self.topology.add_peer(PeerInfo(
                node_id=node_info.node_id,
                hostname=node_info.hostname,
                address=node_info.address,
                port=node_info.port,
                chip=node_info.chip,
                ram_gb=node_info.ram_gb,
                gpu_cores=node_info.gpu_cores,
                available_gb=node_info.available_gb,
                capability_score=_compute_capability_score(node_info),
            ))
            self.topology.update_link(
                local.node_id, node_info.node_id,
                bandwidth_mbps=bw, latency_ms=lat, link_type=link_type,
            )

        num_peers = len(self.topology.active_nodes)
        logger.info("Mesh: %d nodes, %dGB total RAM", num_peers, self.topology.total_ram_gb)

        if num_peers <= 1:
            logger.info("No peers found — running in single-node mode")
            self._is_coordinator = True
            self.node.become_coordinator()
            self.topology.set_coordinator(local.node_id)
            self._emit_status("single_node")
            return await self._load_model_single_node()

        # Step 3: Elect coordinator
        self.node.set_state(NodeState.JOINING)
        self._emit_status("electing")

        election = MeshElection(
            local_node_id=local.node_id,
            local_hostname=local.hostname,
            local_score=self.node.capability_score,
            local_ram_gb=local.ram_gb,
            local_gpu_cores=local.gpu_cores,
            forced_coordinator_id=local.node_id if self.force_coordinator else None,
        )

        # Add votes from discovered peers
        for peer in self.topology.active_nodes:
            if peer.node_id != local.node_id:
                election.receive_vote(
                    peer.node_id, peer.hostname, peer.capability_score,
                    peer.ram_gb, peer.gpu_cores,
                )

        result = await election.start_election()
        self._is_coordinator = (result.coordinator_id == local.node_id)
        self.topology.set_coordinator(result.coordinator_id)

        if self._is_coordinator:
            self.node.become_coordinator()
            logger.info("This node is the COORDINATOR")
        else:
            self.node.accept_coordinator(result.coordinator_id)
            logger.info("Coordinator: %s", result.hostname)

        # Step 4: Set up distributed inference
        self.node.set_state(NodeState.READY)
        self._emit_status("loading")

        if self._is_coordinator:
            success = await self._setup_as_coordinator(discovered)
        else:
            success = await self._setup_as_worker()

        if success:
            self._model_loaded = True
            self.node.set_state(NodeState.ACTIVE)
            self._emit_status("ready")

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())

        return success

    async def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        if self.coordinator:
            await self.coordinator.shutdown_cluster()
        if self.worker:
            await self.worker.shutdown()
        self.node.set_state(NodeState.SHUTDOWN)
        self._emit_status("shutdown")
        logger.info("Mesh shut down")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def forward(self, input_ids: mx.array, cache=None) -> mx.array:
        """Run a distributed forward pass.

        Only callable on the coordinator node.
        """
        if not self._is_coordinator or not self.coordinator:
            raise RuntimeError("forward() can only be called on the coordinator")
        return await self.coordinator.forward(input_ids, cache=cache)

    def get_tokenizer(self):
        if self.coordinator:
            return self.coordinator.tokenizer
        return None

    def get_model(self):
        if self.coordinator:
            return self.coordinator.model
        return None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    async def _setup_as_coordinator(self, discovered: List[NodeInfo]) -> bool:
        """Set up this node as the coordinator."""
        self.coordinator = Coordinator(
            cluster_secret=self.cluster_secret,
            model_path=self.model_path,
        )
        try:
            await self.coordinator.setup_cluster(
                model_path=self.model_path,
                nodes=discovered,
                smelt_percent=self.smelt_percent,
                local_layers=True,
            )
            return True
        except Exception as e:
            logger.error("Failed to set up coordinator: %s", e)
            return False

    async def _setup_as_worker(self) -> bool:
        """Set up this node as a worker (wait for coordinator to assign layers)."""
        self.worker = Worker(
            port=self.node.info.port,
            cluster_secret=self.cluster_secret,
        )
        # Worker serves and waits for coordinator to connect + assign layers
        asyncio.create_task(self.worker.serve())
        logger.info("Worker started, waiting for coordinator to assign layers...")
        return True

    async def _load_model_single_node(self) -> bool:
        """Load model on a single node (no distribution needed)."""
        # In single-node mode, just load normally via the existing engine
        # The distributed flag is a no-op if no peers are found
        logger.info("Single-node mode — model will be loaded by the standard engine")
        self._model_loaded = True
        return True

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self):
        """Send heartbeats to peers via coordinator connections and detect failures."""
        while self._running:
            try:
                # Update our own heartbeat in topology
                self.topology.update_heartbeat(self.node.node_id)

                # If we're coordinator, health-check all workers via their TCP connections
                if self._is_coordinator and self.coordinator:
                    for node_id, wc in list(self.coordinator.workers.items()):
                        try:
                            health = await wc.health_check()
                            if health.get("status") not in ("error", "disconnected"):
                                self.topology.update_heartbeat(node_id)
                            else:
                                logger.debug("Worker %s health check: %s", node_id, health.get("status"))
                        except Exception as e:
                            logger.debug("Heartbeat to %s failed: %s", node_id, e)

                # Check for dead nodes (no heartbeat for DEAD_TIMEOUT seconds)
                dead = self.topology.get_dead_nodes()
                for peer in dead:
                    # Don't declare ourselves dead
                    if peer.node_id == self.node.node_id:
                        continue
                    logger.warning("Node %s DEAD (no heartbeat for %ds)", peer.hostname, DEAD_TIMEOUT)
                    self.topology.remove_peer(peer.node_id)
                    self._emit_topology_change("node_dead", peer.node_id)

                    if peer.is_coordinator and not self._is_coordinator:
                        # Coordinator died! Re-elect
                        logger.critical("Coordinator %s lost! Triggering re-election...", peer.hostname)
                        await self._handle_coordinator_loss()

                # Check for suspect nodes
                for peer in self.topology.get_suspect_nodes():
                    if peer.node_id != self.node.node_id:
                        logger.debug("Node %s SUSPECT", peer.hostname)

                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: %s", e)
                await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _monitor_loop(self):
        """Monitor mesh health and re-measure links periodically."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Re-check every 60s

                # Re-measure bandwidth to all peers
                local_id = self.node.node_id
                for peer in self.topology.active_nodes:
                    if peer.node_id == local_id:
                        continue
                    bw, lat = await measure_bandwidth(peer.address, peer.port)
                    if bw > 0:
                        link_type = detect_best_route(peer.address)
                        self.topology.update_link(
                            local_id, peer.node_id,
                            bandwidth_mbps=bw, latency_ms=lat, link_type=link_type,
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Monitor error: %s", e)

    async def _handle_coordinator_loss(self):
        """Handle coordinator failure — re-elect and redistribute."""
        local = self.node.info
        election = MeshElection(
            local_node_id=local.node_id,
            local_hostname=local.hostname,
            local_score=self.node.capability_score,
            local_ram_gb=local.ram_gb,
            local_gpu_cores=local.gpu_cores,
        )

        for peer in self.topology.active_nodes:
            if peer.node_id != local.node_id:
                election.receive_vote(
                    peer.node_id, peer.hostname, peer.capability_score,
                    peer.ram_gb, peer.gpu_cores,
                )

        result = await election.start_election()
        self._is_coordinator = (result.coordinator_id == local.node_id)
        self.topology.set_coordinator(result.coordinator_id)

        if self._is_coordinator:
            self.node.become_coordinator()
            logger.info("Re-elected as coordinator — redistributing layers...")
            # TODO: redistribute layers from dead coordinator to surviving nodes

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_status(self, status: str):
        if self._on_status_change:
            self._on_status_change(status, self.status)

    def _emit_topology_change(self, event: str, node_id: str):
        if self._on_topology_change:
            self._on_topology_change(event, node_id, self.topology.to_snapshot())


def _compute_capability_score(info: NodeInfo) -> float:
    return (
        info.ram_gb * 0.4
        + info.gpu_cores * 0.3
        + info.available_gb * 0.2
    )

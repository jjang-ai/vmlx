# SPDX-License-Identifier: Apache-2.0
"""MeshTopology — in-memory graph of the compute mesh.

Pure Python, no external graph library. Tracks nodes, connections,
bandwidth measurements, and computes optimal pipeline ordering.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .discovery import LinkType

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Info about a peer node in the mesh."""
    node_id: str
    hostname: str
    address: str
    port: int
    chip: str = ""
    ram_gb: int = 0
    gpu_cores: int = 0
    available_gb: float = 0.0
    capability_score: float = 0.0
    state: str = "discovered"
    is_coordinator: bool = False
    assigned_layers: Optional[Tuple[int, int]] = None
    model_loaded: bool = False
    last_heartbeat: float = field(default_factory=time.time)
    joined_at: float = field(default_factory=time.time)

    @property
    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 30.0

    @property
    def is_suspect(self) -> bool:
        return 15.0 < time.time() - self.last_heartbeat < 30.0


@dataclass
class LinkInfo:
    """Measured connection between two nodes."""
    src_id: str
    dst_id: str
    link_type: LinkType = LinkType.UNKNOWN
    bandwidth_mbps: float = 0.0
    latency_ms: float = 999.0
    last_measured: float = field(default_factory=time.time)

    @property
    def quality_score(self) -> float:
        """Higher = better link. Combines bandwidth and latency."""
        bw_score = min(self.bandwidth_mbps / 1000, 100)  # cap at 100Gbps
        lat_score = max(0, 10 - self.latency_ms)          # 0ms=10, 10ms=0
        return bw_score * 0.7 + lat_score * 0.3


class MeshTopology:
    """In-memory graph of the compute mesh."""

    def __init__(self):
        self.peers: Dict[str, PeerInfo] = {}
        self.links: Dict[str, Dict[str, LinkInfo]] = {}  # src → dst → link
        self._coordinator_id: Optional[str] = None

    @property
    def coordinator(self) -> Optional[PeerInfo]:
        if self._coordinator_id and self._coordinator_id in self.peers:
            return self.peers[self._coordinator_id]
        return None

    @property
    def active_nodes(self) -> List[PeerInfo]:
        return [p for p in self.peers.values() if p.is_alive]

    @property
    def total_ram_gb(self) -> int:
        return sum(p.ram_gb for p in self.active_nodes)

    @property
    def total_available_gb(self) -> float:
        return sum(p.available_gb for p in self.active_nodes)

    def add_peer(self, peer: PeerInfo):
        self.peers[peer.node_id] = peer
        if peer.node_id not in self.links:
            self.links[peer.node_id] = {}
        logger.debug("Topology: added peer %s (%s, %dGB)", peer.hostname, peer.chip, peer.ram_gb)

    def remove_peer(self, node_id: str):
        self.peers.pop(node_id, None)
        self.links.pop(node_id, None)
        for links in self.links.values():
            links.pop(node_id, None)
        if self._coordinator_id == node_id:
            self._coordinator_id = None
        logger.debug("Topology: removed peer %s", node_id)

    def update_heartbeat(self, node_id: str):
        if node_id in self.peers:
            self.peers[node_id].last_heartbeat = time.time()

    def set_coordinator(self, node_id: str):
        self._coordinator_id = node_id
        for peer in self.peers.values():
            peer.is_coordinator = (peer.node_id == node_id)

    def update_link(self, src_id: str, dst_id: str, bandwidth_mbps: float, latency_ms: float, link_type: LinkType = LinkType.UNKNOWN):
        if src_id not in self.links:
            self.links[src_id] = {}
        self.links[src_id][dst_id] = LinkInfo(
            src_id=src_id, dst_id=dst_id,
            link_type=link_type, bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
        )

    def get_link(self, src_id: str, dst_id: str) -> Optional[LinkInfo]:
        return self.links.get(src_id, {}).get(dst_id)

    def get_dead_nodes(self) -> List[PeerInfo]:
        return [p for p in self.peers.values() if not p.is_alive]

    def get_suspect_nodes(self) -> List[PeerInfo]:
        return [p for p in self.peers.values() if p.is_suspect]

    # ------------------------------------------------------------------
    # Pipeline ordering
    # ------------------------------------------------------------------

    def compute_pipeline_order(self) -> List[str]:
        """Compute optimal node ordering for pipeline parallelism.

        Strategy:
        1. If TB5 ring exists → use that (all transfers on fast bus)
        2. Otherwise → greedy: start at coordinator, pick nearest neighbor
        """
        alive = [p for p in self.peers.values() if p.is_alive]
        if len(alive) <= 1:
            return [p.node_id for p in alive]

        # Check for Thunderbolt ring
        tb_ring = self._find_thunderbolt_ring()
        if tb_ring:
            logger.info("Pipeline order: Thunderbolt ring detected (%d nodes)", len(tb_ring))
            return tb_ring

        # Greedy nearest-neighbor from coordinator
        start = self._coordinator_id or alive[0].node_id
        order = [start]
        remaining = {p.node_id for p in alive} - {start}

        while remaining:
            current = order[-1]
            best = None
            best_quality = -1
            for nid in remaining:
                link = self.get_link(current, nid)
                quality = link.quality_score if link else 0
                if quality > best_quality:
                    best_quality = quality
                    best = nid
            if best:
                order.append(best)
                remaining.remove(best)
            else:
                # No measured link — just append remaining
                order.extend(sorted(remaining))
                break

        return order

    def _find_thunderbolt_ring(self) -> Optional[List[str]]:
        """Find a cycle through TB5-connected nodes (if one exists)."""
        tb_neighbors: Dict[str, Set[str]] = {}
        for src_id, dsts in self.links.items():
            for dst_id, link in dsts.items():
                if link.link_type == LinkType.THUNDERBOLT:
                    tb_neighbors.setdefault(src_id, set()).add(dst_id)
                    tb_neighbors.setdefault(dst_id, set()).add(src_id)

        if not tb_neighbors:
            return None

        # Simple ring detection: DFS from any TB node
        # For 2-10 nodes this is trivial
        start = next(iter(tb_neighbors))
        visited = [start]
        current = start

        while True:
            neighbors = tb_neighbors.get(current, set()) - set(visited)
            if not neighbors:
                break
            nxt = next(iter(neighbors))
            visited.append(nxt)
            current = nxt

        # Check if it forms a cycle back to start
        if start in tb_neighbors.get(current, set()) and len(visited) > 1:
            return visited

        # Not a full ring, but still a TB chain
        if len(visited) > 1:
            return visited

        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict:
        return {
            "coordinator_id": self._coordinator_id,
            "peers": {nid: {
                "hostname": p.hostname, "chip": p.chip, "ram_gb": p.ram_gb,
                "gpu_cores": p.gpu_cores, "state": p.state,
                "capability_score": p.capability_score,
                "assigned_layers": p.assigned_layers,
            } for nid, p in self.peers.items()},
            "links": {src: {dst: {
                "link_type": l.link_type.value, "bandwidth_mbps": l.bandwidth_mbps,
                "latency_ms": l.latency_ms,
            } for dst, l in dsts.items()} for src, dsts in self.links.items()},
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "MeshTopology":
        topo = cls()
        topo._coordinator_id = data.get("coordinator_id")
        for nid, info in data.get("peers", {}).items():
            topo.add_peer(PeerInfo(
                node_id=nid, hostname=info["hostname"], address="",
                port=0, chip=info.get("chip", ""),
                ram_gb=info.get("ram_gb", 0), gpu_cores=info.get("gpu_cores", 0),
                capability_score=info.get("capability_score", 0),
                state=info.get("state", "unknown"),
            ))
        for src, dsts in data.get("links", {}).items():
            for dst, linfo in dsts.items():
                topo.update_link(
                    src, dst,
                    bandwidth_mbps=linfo.get("bandwidth_mbps", 0),
                    latency_ms=linfo.get("latency_ms", 999),
                    link_type=LinkType(linfo.get("link_type", "unknown")),
                )
        return topo

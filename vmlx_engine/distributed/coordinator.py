# SPDX-License-Identifier: Apache-2.0
"""Distributed inference coordinator.

The coordinator runs on the primary Mac (which also runs the vMLX Panel).
It manages worker nodes, assigns layers, orchestrates forward passes,
and handles tokenization + sampling. Workers only see hidden state tensors.

The coordinator holds:
- Embedding layer (embed_tokens)
- Final projection (lm_head / as_linear)
- Its own assigned layer range (if any)
- Tokenizer
- KV cache coordinator state
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import mlx.core as mx

from .discovery import (
    BonjourScanner,
    NodeInfo,
    detect_best_route,
    measure_bandwidth,
)
from .layer_assign import (
    NodeAssignment,
    assign_layers_by_ram,
    estimate_layer_profiles,
)
from .protocol import (
    Message,
    MessageType,
    deserialize_tensor,
    make_forward,
    make_health,
    make_join,
    make_load_layers,
    make_shutdown,
    serialize_tensor,
)

logger = logging.getLogger(__name__)


class WorkerConnection:
    """Persistent connection to a worker node."""

    def __init__(self, node: NodeInfo):
        self.node = node
        self.assignment: Optional[NodeAssignment] = None
        self._reader = None
        self._writer = None
        self._connected = False

    async def connect(self, cluster_secret: str) -> bool:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.node.address, self.node.port),
                timeout=10.0,
            )
            # Authenticate
            join_msg = make_join(cluster_secret, self.node.to_dict())
            self._writer.write(join_msg.encode())
            await self._writer.drain()

            resp = await Message.read_from(self._reader)
            if resp.type == MessageType.JOIN_ACK and resp.metadata.get("accepted"):
                self._connected = True
                caps = resp.metadata.get("capabilities", {})
                self.node.chip = caps.get("chip", self.node.chip)
                self.node.ram_gb = caps.get("ram_gb", self.node.ram_gb)
                self.node.gpu_cores = caps.get("gpu_cores", self.node.gpu_cores)
                self.node.available_gb = caps.get("available_gb", self.node.available_gb)
                self.node.status = "connected"
                logger.info("Connected to worker %s (%s, %dGB)", self.node.hostname, self.node.chip, self.node.ram_gb)
                return True
            else:
                reason = resp.metadata.get("reason", "unknown")
                logger.warning("Worker %s rejected join: %s", self.node.hostname, reason)
                return False
        except Exception as e:
            logger.error("Failed to connect to worker %s: %s", self.node.hostname, e)
            return False

    async def load_layers(self, model_path: str, start: int, end: int, quantization: dict = None) -> bool:
        if not self._connected:
            return False
        msg = make_load_layers(model_path, start, end, quantization)
        self._writer.write(msg.encode())
        await self._writer.drain()

        resp = await asyncio.wait_for(Message.read_from(self._reader), timeout=300.0)
        if resp.type == MessageType.LOAD_ACK and resp.metadata.get("success"):
            self.node.status = "ready"
            self.node.assigned_layers = (start, end)
            logger.info(
                "Worker %s loaded layers %d-%d (%.1fGB used, %.1fGB free)",
                self.node.hostname, start, end - 1,
                resp.metadata.get("memory_used_gb", 0),
                resp.metadata.get("memory_available_gb", 0),
            )
            return True
        logger.error("Worker %s failed to load layers %d-%d", self.node.hostname, start, end - 1)
        return False

    async def forward(self, hidden: mx.array, request_id: str, seq_pos: int, cache_id: str = None) -> mx.array:
        if not self._connected:
            raise ConnectionError(f"Worker {self.node.hostname} not connected")

        msg = make_forward(hidden, request_id, seq_pos, cache_id)
        self._writer.write(msg.encode())
        await self._writer.drain()

        resp = await Message.read_from(self._reader)
        if resp.type == MessageType.FORWARD_RESULT:
            compute_ms = resp.metadata.get("compute_time_ms", 0)
            logger.debug(
                "Worker %s forward: %.1fms", self.node.hostname, compute_ms,
            )
            return deserialize_tensor(resp.payload)
        elif resp.type == MessageType.ERROR:
            raise RuntimeError(f"Worker {self.node.hostname} error: {resp.metadata.get('message', 'unknown')}")
        raise RuntimeError(f"Unexpected response type: {resp.type}")

    async def health_check(self) -> dict:
        if not self._connected:
            return {"status": "disconnected"}
        self._writer.write(make_health().encode())
        await self._writer.drain()
        resp = await asyncio.wait_for(Message.read_from(self._reader), timeout=5.0)
        return resp.metadata if resp.type == MessageType.HEALTH_ACK else {"status": "error"}

    async def disconnect(self):
        if self._connected:
            try:
                self._writer.write(make_shutdown().encode())
                await self._writer.drain()
            except Exception:
                pass
            self._writer.close()
            await self._writer.wait_closed()
            self._connected = False


class Coordinator:
    """Orchestrates distributed inference across worker nodes.

    Pipeline parallelism flow for each token:
    1. Coordinator: embed_tokens(input_ids) → hidden
    2. Coordinator: forward through local layers (if any)
    3. Send hidden → Worker 1 → hidden → Worker 2 → ... → hidden back
    4. Coordinator: lm_head(hidden) → logits
    5. Coordinator: sample → next token
    """

    def __init__(
        self,
        cluster_secret: str = "",
        model_path: str = "",
    ):
        self.cluster_secret = cluster_secret
        self.model_path = model_path
        self.workers: Dict[str, WorkerConnection] = {}
        self.assignments: List[NodeAssignment] = []
        self.pipeline_order: List[str] = []  # node_ids in forward-pass order

        self.model = None
        self.tokenizer = None
        self.local_layers = None
        self.local_assignment: Optional[NodeAssignment] = None

        self._scanner = BonjourScanner()

    async def discover_nodes(self, timeout: float = 5.0) -> List[NodeInfo]:
        """Scan for available worker nodes."""
        nodes = await self._scanner.scan(timeout)
        logger.info("Discovered %d worker nodes", len(nodes))
        for node in nodes:
            bw, lat = await measure_bandwidth(node.address, node.port)
            node.measured_bandwidth_mbps = bw
            node.measured_latency_ms = lat
            node.link_type = detect_best_route(node.address)
            logger.info(
                "  %s: %s, %dGB RAM, %.0f Mbps, %.1fms latency (%s)",
                node.hostname, node.chip, node.ram_gb,
                bw, lat, node.link_type.value,
            )
        return nodes

    async def add_node_manual(self, address: str, port: int = 9100) -> Optional[NodeInfo]:
        """Manually add a worker node by IP address."""
        node = NodeInfo(
            node_id=f"manual-{address}:{port}",
            hostname=address,
            address=address,
            port=port,
            status="discovered",
        )
        bw, lat = await measure_bandwidth(address, port)
        node.measured_bandwidth_mbps = bw
        node.measured_latency_ms = lat
        node.link_type = detect_best_route(address)

        wc = WorkerConnection(node)
        if await wc.connect(self.cluster_secret):
            self.workers[node.node_id] = wc
            return wc.node
        return None

    async def setup_cluster(
        self,
        model_path: str,
        nodes: List[NodeInfo],
        smelt_percent: int = 100,
        local_layers: bool = True,
    ):
        """Set up the distributed cluster: connect, assign layers, load.

        Args:
            model_path: Path to the JANG model directory.
            nodes: Worker nodes to include (from discover or manual add).
            smelt_percent: Smelt expert loading percentage (100 = all).
            local_layers: Whether the coordinator also runs layers.
        """
        import json
        config = json.loads(open(f"{model_path}/config.json").read())
        text_cfg = config.get("text_config", config)
        num_layers = text_cfg.get("num_hidden_layers", 0)

        if num_layers == 0:
            raise ValueError("Could not determine num_hidden_layers from config")

        # Estimate total weight size from safetensors
        total_gb = _estimate_model_size_gb(model_path)

        # Build layer profiles
        profiles = estimate_layer_profiles(config, total_gb, num_layers, smelt_percent)

        # Build node list for assignment (including local if requested)
        from .discovery import get_local_node_info
        assignment_nodes = []
        if local_layers:
            local = get_local_node_info()
            assignment_nodes.append({
                "node_id": "coordinator",
                "hostname": local.hostname,
                "available_gb": local.available_gb,
                "relative_compute": 1.0,
            })

        # Connect to workers
        for node in nodes:
            wc = WorkerConnection(node)
            if await wc.connect(self.cluster_secret):
                self.workers[node.node_id] = wc
                assignment_nodes.append({
                    "node_id": node.node_id,
                    "hostname": node.hostname,
                    "available_gb": node.available_gb,
                    "relative_compute": 1.0,
                })

        if not assignment_nodes:
            raise RuntimeError("No nodes available for layer assignment")

        # Assign layers
        logger.info("Assigning %d layers across %d nodes:", num_layers, len(assignment_nodes))
        self.assignments = assign_layers_by_ram(profiles, assignment_nodes)

        # Load model on coordinator — only our layer range + embed/lm_head
        # Non-layer weights (embed_tokens, lm_head, norms) are always kept by _filter_by_layer_range
        from vmlx_engine.utils.jang_loader import load_jang_model

        if local_layers and self.assignments:
            self.local_assignment = self.assignments[0]  # coordinator is first
            layer_range = (self.local_assignment.layer_start, self.local_assignment.layer_end)
            logger.info(
                "Loading coordinator model (layers %d-%d + embed/lm_head)...",
                layer_range[0], layer_range[1] - 1,
            )
            self.model, self.tokenizer = load_jang_model(
                model_path, layer_range=layer_range,
            )
            all_layers = _get_layers_list(self.model)
            self.local_layers = list(
                all_layers[self.local_assignment.layer_start:self.local_assignment.layer_end]
            )
        else:
            # No local layers — just load embed/lm_head (layer_range=(0,0) loads no layers)
            logger.info("Loading coordinator model (embed/lm_head only)...")
            self.model, self.tokenizer = load_jang_model(
                model_path, layer_range=(0, 0),
            )

        # Load layers on workers
        for assignment in self.assignments:
            if assignment.node_id == "coordinator":
                continue
            wc = self.workers.get(assignment.node_id)
            if wc:
                success = await wc.load_layers(
                    model_path, assignment.layer_start, assignment.layer_end,
                )
                if not success:
                    raise RuntimeError(f"Worker {assignment.hostname} failed to load layers")

        # Build pipeline order
        self.pipeline_order = [a.node_id for a in self.assignments]
        logger.info("Cluster ready: %s", " → ".join(
            f"{a.hostname}[L{a.layer_start}-{a.layer_end-1}]" for a in self.assignments
        ))

    async def forward(self, input_ids: mx.array, cache=None) -> mx.array:
        """Run a distributed forward pass.

        1. Embed on coordinator
        2. Pipeline through nodes in order
        3. lm_head on coordinator
        """
        # Embedding
        hidden = self.model.model.embed_tokens(input_ids)

        # Pipeline through each node's layer range
        for assignment in self.assignments:
            if assignment.node_id == "coordinator":
                # Run local layers
                if self.local_layers:
                    for layer in self.local_layers:
                        hidden = layer(hidden, mask=None, cache=None)
            else:
                # Send to worker
                wc = self.workers[assignment.node_id]
                hidden = await wc.forward(
                    hidden, request_id="", seq_pos=0,
                )

        # lm_head
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(hidden)
        elif hasattr(self.model.model, "embed_tokens"):
            logits = self.model.model.embed_tokens.as_linear(hidden)
        else:
            logits = hidden

        return logits

    async def shutdown_cluster(self):
        for wc in self.workers.values():
            await wc.disconnect()
        self.workers.clear()
        logger.info("Cluster shut down")


def _get_layers_list(model):
    for accessor in [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.backbone.layers,
    ]:
        try:
            return accessor(model)
        except AttributeError:
            continue
    raise ValueError("Could not find model layers")


def _estimate_model_size_gb(model_path: str) -> float:
    import glob
    import os
    total = sum(
        os.path.getsize(f)
        for f in glob.glob(f"{model_path}/*.safetensors")
    )
    return total / (1024 ** 3)

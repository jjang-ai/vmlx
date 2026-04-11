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

    async def forward(
        self,
        hidden: mx.array,
        request_id: str,
        seq_pos: int,
    ) -> mx.array:
        if not self._connected:
            raise ConnectionError(f"Worker {self.node.hostname} not connected")
        if not request_id:
            raise ValueError("request_id must be non-empty")

        msg = make_forward(hidden, request_id, seq_pos, cache_id=request_id)
        self._writer.write(msg.encode())
        await self._writer.drain()

        # Bounded wait so a wedged worker can't hang the coordinator forever.
        # 120s is generous for very large prefills on slow links; decode steps
        # typically return in single-digit ms.
        resp = await asyncio.wait_for(
            Message.read_from(self._reader), timeout=120.0,
        )
        if resp.type == MessageType.FORWARD_RESULT:
            compute_ms = resp.metadata.get("compute_time_ms", 0)
            logger.debug(
                "Worker %s forward: %.1fms", self.node.hostname, compute_ms,
            )
            return deserialize_tensor(resp.payload)
        elif resp.type == MessageType.ERROR:
            raise RuntimeError(
                f"Worker {self.node.hostname} error: "
                f"{resp.metadata.get('message', 'unknown')}"
            )
        raise RuntimeError(f"Unexpected response type: {resp.type}")

    async def release_cache(self, request_id: str) -> bool:
        """Drop the KV cache slot for a finished request on this worker.

        Fire-and-forget in effect — we wait briefly for the ack but treat
        failures as non-fatal (the worker will GC the slot when the process
        ends, and a stuck worker isn't going to unwedge itself here).
        """
        if not self._connected:
            return False
        from .protocol import make_cache_release
        try:
            self._writer.write(make_cache_release(request_id).encode())
            await self._writer.drain()
            resp = await asyncio.wait_for(
                Message.read_from(self._reader), timeout=5.0,
            )
            return resp.type == MessageType.CACHE_ACK and resp.metadata.get("success", False)
        except Exception as e:
            logger.debug(
                "Worker %s cache release for %s failed: %s",
                self.node.hostname, request_id, e,
            )
            return False

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
        with open(f"{model_path}/config.json") as _f:
            config = json.loads(_f.read())
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

    def make_local_cache(self) -> Optional[list]:
        """Build a KV cache for the coordinator's own layer slice.

        Each active request gets its own local cache object so concurrent
        requests don't corrupt each other. Returns None when:
          - the coordinator owns no local layers, or
          - the model's make_cache() returns a list whose length doesn't
            match the full layer count (hybrid SSM case — Nemotron only
            creates cache entries for attention + SSM layers, skipping
            MLP/MoE. A naive slice would feed the wrong cached K/V to the
            wrong layer). Returning None forces layers to run without
            cache (correct but O(N²) — see the Phase 2 TODO).
        """
        if not self.local_layers or not self.local_assignment:
            return None
        try:
            raw_model = self.model
            for accessor in (
                lambda m: m.language_model.model,
                lambda m: m.model,
                lambda m: m.backbone,
            ):
                try:
                    raw_model = accessor(self.model)
                    break
                except AttributeError:
                    continue
            if hasattr(raw_model, "make_cache"):
                full_cache = raw_model.make_cache()
            elif hasattr(self.model, "make_cache"):
                full_cache = self.model.make_cache()
            else:
                from mlx_lm.models.cache import make_prompt_cache
                full_cache = make_prompt_cache(self.model)

            if not isinstance(full_cache, list):
                return None

            # Only slice when we can establish a 1:1 correspondence between
            # cache entries and layers. For hybrid SSM models, the cache
            # length is less than the layer count because MLP/MoE layers
            # have no cache entries — slicing by layer index would be
            # incorrect. Bail out with a warning in that case.
            try:
                all_layers = _get_layers_list(self.model)
                num_layers = len(all_layers)
            except ValueError:
                num_layers = None

            if num_layers is not None and len(full_cache) != num_layers:
                logger.warning(
                    "Coordinator cache slice skipped: model has %d layers "
                    "but make_cache() returned %d entries (hybrid SSM?). "
                    "Running local layers without cache — generation will "
                    "be O(N²). This is a known Phase 2 limitation for "
                    "Nemotron-style hybrid models in distributed mode.",
                    num_layers, len(full_cache),
                )
                return None

            start = self.local_assignment.layer_start
            end = self.local_assignment.layer_end
            if len(full_cache) < end:
                logger.warning(
                    "Coordinator cache slice [%d, %d) out of range for "
                    "cache of length %d — running local layers without cache.",
                    start, end, len(full_cache),
                )
                return None
            return full_cache[start:end]
        except Exception as e:
            logger.warning(
                "Could not create local cache for layers %d-%d: %s — "
                "running coordinator layers without cache (slow)",
                self.local_assignment.layer_start,
                self.local_assignment.layer_end - 1, e,
            )
            return None

    async def forward(
        self,
        input_ids: mx.array,
        request_id: str,
        seq_pos: int,
        local_cache: Optional[list] = None,
    ) -> mx.array:
        """Run a distributed forward pass.

        Args:
            input_ids: [1, seq_len] token ids (prefill) or [1, 1] (decode).
            request_id: Unique id used to key per-request KV cache slots on
                each worker. Must be non-empty.
            seq_pos: 0-indexed position of the first token in input_ids
                within the full sequence. Prefill passes 0; decode passes
                prompt_len + generated_count - 1.
            local_cache: KV cache for the coordinator's own layer slice. See
                make_local_cache(). None means "no cache" (correct only for
                single-token-only inference, which is the pre-fix behavior).

        1. Embed on coordinator
        2. Pipeline through coordinator local layers (with cache) then workers
        3. lm_head on coordinator
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")

        # Multi-accessor embed_tokens fallback — different model families
        # expose embeddings at different paths:
        #   - Qwen / Mistral / Gemma / Llama text:  model.model.embed_tokens
        #   - VLM wrappers (Qwen3.5-VL, Gemma 4 VL): model.language_model.model.embed_tokens
        #   - Nemotron hybrid:                       model.backbone.embeddings
        # Previous hardcoded `self.model.model.embed_tokens` crashed Nemotron
        # with AttributeError on the very first token.
        _embed = None
        for _accessor in (
            lambda m: m.language_model.model.embed_tokens,
            lambda m: m.model.embed_tokens,
            lambda m: m.backbone.embeddings,
            lambda m: m.backbone.embed_tokens,
        ):
            try:
                _embed = _accessor(self.model)
                break
            except AttributeError:
                continue
        if _embed is None:
            raise RuntimeError(
                "Could not locate embed_tokens on the model. Distributed "
                "inference needs to embed input tokens on the coordinator. "
                "Model: " + type(self.model).__name__
            )
        hidden = _embed(input_ids)

        seq_len = hidden.shape[1]
        local_mask = _build_causal_mask(seq_len, seq_pos)

        # Pipeline through each node's layer range in assignment order.
        for assignment in self.assignments:
            if assignment.node_id == "coordinator":
                if self.local_layers:
                    for i, layer in enumerate(self.local_layers):
                        c = local_cache[i] if local_cache else None
                        hidden = layer(hidden, mask=local_mask, cache=c)
            else:
                wc = self.workers[assignment.node_id]
                hidden = await wc.forward(
                    hidden, request_id=request_id, seq_pos=seq_pos,
                )

        # lm_head or tied-embedding projection. Multi-accessor so backbone
        # and VLM-wrapped models work — the old hardcoded path assumed
        # self.model.lm_head OR self.model.model.embed_tokens.
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            logits = self.model.lm_head(hidden)
        else:
            logits = None
            for _accessor in (
                lambda m: m.language_model.model.embed_tokens,
                lambda m: m.model.embed_tokens,
                lambda m: m.backbone.embeddings,
                lambda m: m.backbone.embed_tokens,
            ):
                try:
                    _embed = _accessor(self.model)
                    if _embed is not None and hasattr(_embed, "as_linear"):
                        logits = _embed.as_linear(hidden)
                        break
                except AttributeError:
                    continue
            if logits is None:
                logits = hidden  # no projection available — return raw hidden
        return logits

    async def release_request(self, request_id: str) -> None:
        """Drop KV cache slots for a finished request on every worker.

        Called from DistributedEngine's generate loop `finally` block so
        workers don't accumulate stale caches across requests.
        """
        if not request_id:
            return
        for wc in self.workers.values():
            try:
                await wc.release_cache(request_id)
            except Exception as e:
                logger.debug(
                    "release_cache failed on %s: %s", wc.node.hostname, e,
                )

    async def shutdown_cluster(self):
        for wc in self.workers.values():
            await wc.disconnect()
        self.workers.clear()
        logger.info("Cluster shut down")


def _build_causal_mask(query_len: int, seq_pos: int) -> Optional[mx.array]:
    """Build a causal attention mask for the coordinator's local layers.

    Mirrors the worker's `_create_attention_mask` but runs in the coordinator
    process so local-layer forwards get correct masking too. Returns None for
    single-token decode (MLX models handle unmasked 1-token forward).
    """
    if query_len == 1:
        return None
    import numpy as np
    total_seq = seq_pos + query_len
    mask_np = np.full((query_len, total_seq), -1e9, dtype=np.float32)
    offset = total_seq - query_len
    for i in range(query_len):
        mask_np[i, : offset + i + 1] = 0.0
    return mx.array(mask_np).reshape(1, 1, query_len, total_seq)


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

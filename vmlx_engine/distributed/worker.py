# SPDX-License-Identifier: Apache-2.0
"""Distributed inference worker node.

A worker loads a subset of transformer layers and runs forward passes
on hidden states received from the coordinator or upstream worker.

Usage:
    vmlx-worker --port 9100 --secret <cluster-secret>
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import mlx.core as mx

from .discovery import (
    BonjourAdvertiser,
    NodeInfo,
    get_local_node_info,
    DEFAULT_WORKER_PORT,
)
from .protocol import (
    Message,
    MessageType,
    deserialize_tensor,
    make_error,
    make_forward_result,
    make_health_ack,
    make_join_ack,
    make_load_ack,
    serialize_tensor,
)

logger = logging.getLogger(__name__)


class Worker:
    """Distributed inference worker."""

    def __init__(
        self,
        port: int = DEFAULT_WORKER_PORT,
        cluster_secret: str = "",
        advertise: bool = True,
    ):
        self.port = port
        self.cluster_secret = cluster_secret
        self.advertise = advertise

        self.node_info = get_local_node_info()
        self.node_info.port = port

        self.model = None
        self.layers = None
        self.layer_start = 0
        self.layer_end = 0
        self.cache = {}

        self._server = None
        self._advertiser = None
        self._authenticated = False
        self._requests_processed = 0
        self._total_compute_ms = 0.0

    async def serve(self):
        self._server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", self.port,
        )
        logger.info("Worker listening on port %d", self.port)

        # HTTP identity endpoint for discovery probes (GET /node_id)
        self._http_server = await asyncio.start_server(
            self._handle_http, "0.0.0.0", self.port + 1,
        )
        logger.info("HTTP identity endpoint on port %d", self.port + 1)

        # UDP discovery responder
        self._udp_task = asyncio.create_task(self._udp_responder())

        if self.advertise:
            self._advertiser = BonjourAdvertiser(self.node_info)
            await self._advertiser.start()

        async with self._server:
            await self._server.serve_forever()

    async def shutdown(self):
        if self._advertiser:
            await self._advertiser.stop()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Worker shut down")

    async def _handle_connection(self, reader, writer):
        peer = writer.get_extra_info("peername")
        logger.info("Connection from %s", peer)
        try:
            while True:
                msg = await Message.read_from(reader)
                response = await self._dispatch(msg)
                if response:
                    writer.write(response.encode())
                    await writer.drain()
                if msg.type == MessageType.SHUTDOWN:
                    break
        except asyncio.IncompleteReadError:
            logger.info("Connection closed by %s", peer)
        except Exception as e:
            logger.error("Error handling connection from %s: %s", peer, e)
            try:
                err = make_error("internal", str(e))
                writer.write(err.encode())
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, msg: Message) -> Optional[Message]:
        handlers = {
            MessageType.JOIN: self._handle_join,
            MessageType.LOAD_LAYERS: self._handle_load_layers,
            MessageType.FORWARD: self._handle_forward,
            MessageType.HEALTH: self._handle_health,
            MessageType.BANDWIDTH_PROBE: self._handle_bandwidth_probe,
            MessageType.SHUTDOWN: self._handle_shutdown,
        }
        handler = handlers.get(msg.type)
        if handler:
            return await handler(msg)
        return make_error("unknown_type", f"Unknown message type: {msg.type}")

    async def _handle_join(self, msg: Message) -> Message:
        secret = msg.metadata.get("secret", "")
        if self.cluster_secret and secret != self.cluster_secret:
            logger.warning("Join rejected: invalid cluster secret")
            return make_join_ack(False, {"reason": "invalid_secret"})

        self._authenticated = True
        self.node_info.status = "connected"
        caps = self.node_info.to_dict()
        caps["max_layers"] = _estimate_max_layers(self.node_info.available_gb)
        logger.info("Join accepted from coordinator")
        return make_join_ack(True, caps)

    async def _handle_load_layers(self, msg: Message) -> Message:
        if not self._authenticated:
            return make_error("auth", "Not authenticated — send JOIN first")

        model_path = msg.metadata["model_path"]
        self.layer_start = msg.metadata["layer_start"]
        self.layer_end = msg.metadata["layer_end"]

        logger.info(
            "Loading layers %d-%d from %s",
            self.layer_start, self.layer_end - 1, model_path,
        )

        try:
            t0 = time.perf_counter()
            self.model, self.layers = await asyncio.to_thread(
                _load_layer_range, model_path, self.layer_start, self.layer_end,
                msg.metadata.get("quantization"),
            )
            elapsed = time.perf_counter() - t0

            self.node_info.status = "ready"
            self.node_info.assigned_layers = (self.layer_start, self.layer_end)

            mem_used = _get_gpu_memory_gb()
            logger.info(
                "Loaded %d layers in %.1fs (%.1fGB GPU memory)",
                self.layer_end - self.layer_start, elapsed, mem_used,
            )

            return make_load_ack(
                success=True,
                layers_loaded=self.layer_end - self.layer_start,
                memory_used_gb=mem_used,
                memory_available_gb=self.node_info.available_gb - mem_used,
            )
        except Exception as e:
            logger.error("Failed to load layers: %s", e)
            self.node_info.status = "error"
            return make_load_ack(
                success=False, layers_loaded=0,
                memory_used_gb=0, memory_available_gb=self.node_info.available_gb,
            )

    async def _handle_forward(self, msg: Message) -> Message:
        if self.layers is None:
            return make_error("not_loaded", "No layers loaded")

        request_id = msg.metadata.get("request_id", "")
        hidden = deserialize_tensor(msg.payload)

        t0 = time.perf_counter()

        # KV cache management
        cache_id = msg.metadata.get("cache_id", request_id)
        if cache_id not in self.cache:
            # Create KV cache for our layer range
            self.cache[cache_id] = _make_layer_cache(self.model, self.layer_start, self.layer_end)

        layer_cache = self.cache[cache_id]

        # Build attention mask from sequence length
        seq_len = hidden.shape[1]
        total_seq = msg.metadata.get("seq_pos", 0) + seq_len
        mask = _create_attention_mask(seq_len, total_seq, cache=layer_cache)

        # Forward through each layer
        for i, layer in enumerate(self.layers):
            c = layer_cache[i] if layer_cache else None
            hidden = layer(hidden, mask=mask, cache=c)

        mx.async_eval(hidden)
        compute_ms = (time.perf_counter() - t0) * 1000

        self._requests_processed += 1
        self._total_compute_ms += compute_ms

        return make_forward_result(
            hidden_states=hidden,
            request_id=request_id,
            compute_time_ms=compute_ms,
        )

    async def _handle_health(self, msg: Message) -> Message:
        return make_health_ack({
            "status": self.node_info.status,
            "layers": f"{self.layer_start}-{self.layer_end - 1}" if self.layers else "none",
            "requests_processed": self._requests_processed,
            "avg_compute_ms": (
                self._total_compute_ms / self._requests_processed
                if self._requests_processed > 0 else 0
            ),
            "gpu_memory_gb": _get_gpu_memory_gb(),
            "chip": self.node_info.chip,
            "ram_gb": self.node_info.ram_gb,
        })

    async def _handle_bandwidth_probe(self, msg: Message) -> Message:
        return Message(
            type=MessageType.BANDWIDTH_PROBE,
            metadata={"size": msg.payload_size},
            payload=msg.payload,
        )

    async def _handle_shutdown(self, msg: Message) -> Optional[Message]:
        logger.info("Shutdown requested by coordinator")
        self.node_info.status = "stopped"
        if self._advertiser:
            await self._advertiser.stop()
        return None

    # ------------------------------------------------------------------
    # HTTP identity endpoint (for discovery probes)
    # ------------------------------------------------------------------

    async def _handle_http(self, reader, writer):
        """Handle HTTP GET /node_id for discovery probes."""
        try:
            request = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            req_str = request.decode("utf-8", errors="ignore")

            if "GET /node_id" in req_str:
                import json as _json
                body = _json.dumps(self.node_info.to_dict())
                response = (
                    f"HTTP/1.0 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n{body}"
                )
                writer.write(response.encode())
            else:
                writer.write(b"HTTP/1.0 404 Not Found\r\n\r\n")

            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    # ------------------------------------------------------------------
    # UDP discovery responder
    # ------------------------------------------------------------------

    async def _udp_responder(self):
        """Listen for UDP broadcast probes and respond."""
        import socket as _socket
        from .discovery import UDP_DISCOVERY_PORT, UDP_MAGIC

        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        # SO_REUSEPORT allows multiple workers on same machine (e.g. coordinator + worker testing)
        try:
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # Not available on all platforms
        try:
            sock.bind(("", UDP_DISCOVERY_PORT))
        except OSError:
            logger.debug("UDP discovery port %d in use, skipping responder", UDP_DISCOVERY_PORT)
            return

        sock.settimeout(1.0)
        logger.debug("UDP discovery responder listening on port %d", UDP_DISCOVERY_PORT)

        while True:
            try:
                data, addr = sock.recvfrom(4096)
                if UDP_MAGIC in data:
                    # Respond with our info
                    import json as _json
                    response = _json.dumps({
                        "magic": UDP_MAGIC.decode(),
                        "node_id": self.node_info.node_id,
                        "hostname": self.node_info.hostname,
                        "port": self.port,
                        "chip": self.node_info.chip,
                        "ram_gb": self.node_info.ram_gb,
                        "gpu_cores": self.node_info.gpu_cores,
                        "available_gb": self.node_info.available_gb,
                        "vmlx_version": self.node_info.vmlx_version,
                    }).encode()
                    sock.sendto(response, addr)
            except TimeoutError:
                pass
            except Exception:
                pass
            await asyncio.sleep(0)


def _load_layer_range(model_path: str, start: int, end: int, quantization=None):
    """Load only the specified layer range from a model.

    Uses jang_loader's layer_range parameter for efficient selective
    mmap — only safetensors weights for layers [start, end) are loaded.
    Non-layer weights (embed, norms) are loaded but unused layers stay lazy.
    """
    from vmlx_engine.utils.jang_loader import load_jang_model

    model, _tokenizer = load_jang_model(
        model_path, layer_range=(start, end),
    )

    layers_list = None
    for accessor in [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.backbone.layers,
    ]:
        try:
            layers_list = accessor(model)
            break
        except AttributeError:
            continue

    if layers_list is None:
        raise ValueError("Could not find model layers")

    selected = list(layers_list[start:end])
    return model, selected


def _make_layer_cache(model, layer_start: int, layer_end: int) -> list:
    """Create cache entries only for this worker's layer range.

    Uses the model's make_cache() to get the full cache structure,
    then extracts entries for our assigned layers. Handles hybrid SSM
    models where cache entries may be mixed types (KVCache + MambaCache).
    """
    try:
        # Get the raw model (unwrap VLM wrapper if needed)
        raw_model = model
        for accessor in [
            lambda m: m.language_model.model,
            lambda m: m.model,
        ]:
            try:
                raw_model = accessor(model)
                break
            except AttributeError:
                continue

        if hasattr(raw_model, "make_cache"):
            full_cache = raw_model.make_cache()
        elif hasattr(model, "make_cache"):
            full_cache = model.make_cache()
        else:
            from mlx_lm.models.cache import make_prompt_cache
            full_cache = make_prompt_cache(model)

        # Extract only our layer range — preserves cache type per layer
        # (KVCache for attention layers, MambaCache for SSM layers)
        if isinstance(full_cache, list) and len(full_cache) >= layer_end:
            return full_cache[layer_start:layer_end]
        elif isinstance(full_cache, list) and len(full_cache) > 0:
            # Cache length doesn't match layer count (some models share cache entries)
            # Take what we can
            num_cache = len(full_cache)
            ratio = num_cache / max(1, layer_end)
            cache_start = int(layer_start * ratio)
            cache_end = int(layer_end * ratio)
            return full_cache[cache_start:cache_end]
        return full_cache
    except Exception as e:
        logger.warning("Could not create cache for layers %d-%d: %s — running without cache", layer_start, layer_end - 1, e)
        return None


def _create_attention_mask(query_len: int, total_seq_len: int, cache=None) -> mx.array:
    """Create causal attention mask for the current forward pass.

    The mask accounts for:
    - Causal (lower-triangular) masking
    - KV cache length (total_seq_len includes cached tokens)
    """
    if query_len == 1:
        # Decode step: single token attends to all previous
        return None  # Most MLX models handle single-token decode without explicit mask

    # Prefill: causal mask using MLX's additive mask convention
    # Create a lower-triangular mask where future positions are -inf
    import numpy as np
    mask_np = np.full((query_len, total_seq_len), -1e9, dtype=np.float32)
    offset = total_seq_len - query_len
    for i in range(query_len):
        mask_np[i, : offset + i + 1] = 0.0

    mask = mx.array(mask_np).reshape(1, 1, query_len, total_seq_len)
    return mask


def _estimate_max_layers(available_gb: float) -> int:
    return max(1, int(available_gb / 0.5))


def _get_gpu_memory_gb() -> float:
    try:
        return mx.get_active_memory() / (1024 ** 3)
    except Exception:
        return 0.0

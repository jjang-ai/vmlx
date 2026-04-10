# SPDX-License-Identifier: Apache-2.0
"""Wire protocol for distributed inference.

Length-prefixed JSON header + binary payload. All messages follow:

    [4 bytes: header_len (big-endian)] [header_len bytes: JSON] [payload bytes]

Header always contains {"type": "...", "size": <payload_size>}.
Payload is raw bytes (typically mx.array serialized via mx.save/mx.load).

Message types:
    Coordinator → Worker:
        join          — Authenticate + assign to cluster
        load_layers   — Load specific layer range from model
        forward       — Run forward pass on hidden states
        cache_op      — KV cache management (create, extend, trim)
        health        — Health check ping
        shutdown      — Graceful shutdown

    Worker → Coordinator:
        join_ack      — Accept join with capability report
        load_ack      — Layers loaded, memory stats
        forward_result — Hidden states after layer range
        cache_ack     — Cache operation result
        health_ack    — Health response with stats
        error         — Error report

    Bidirectional:
        bandwidth_probe — Bandwidth measurement echo
"""

from __future__ import annotations

import io
import json
import struct
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np


class MessageType(Enum):
    # Coordinator → Worker
    JOIN = "join"
    LOAD_LAYERS = "load_layers"
    FORWARD = "forward"
    CACHE_OP = "cache_op"
    HEALTH = "health"
    SHUTDOWN = "shutdown"

    # Worker → Coordinator
    JOIN_ACK = "join_ack"
    LOAD_ACK = "load_ack"
    FORWARD_RESULT = "forward_result"
    CACHE_ACK = "cache_ack"
    HEALTH_ACK = "health_ack"
    ERROR = "error"

    # Bidirectional
    BANDWIDTH_PROBE = "bandwidth_probe"


@dataclass
class Message:
    """A protocol message with JSON header and optional binary payload."""
    type: MessageType
    metadata: Dict[str, Any]
    payload: Optional[bytes] = None

    @property
    def payload_size(self) -> int:
        return len(self.payload) if self.payload else 0

    def encode(self) -> bytes:
        """Serialize to wire format."""
        header = {
            "type": self.type.value,
            "size": self.payload_size,
            "ts": time.time(),
            **self.metadata,
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        parts = [
            struct.pack(">I", len(header_bytes)),
            header_bytes,
        ]
        if self.payload:
            parts.append(self.payload)
        return b"".join(parts)

    @classmethod
    async def read_from(cls, reader) -> "Message":
        """Read a message from an asyncio StreamReader."""
        hdr_len_bytes = await reader.readexactly(4)
        hdr_len = struct.unpack(">I", hdr_len_bytes)[0]
        if hdr_len > 100 * 1024 * 1024:  # 100MB header limit
            raise ValueError(f"Header too large: {hdr_len}")

        hdr_bytes = await reader.readexactly(hdr_len)
        header = json.loads(hdr_bytes)

        msg_type = MessageType(header.pop("type"))
        payload_size = header.pop("size", 0)
        header.pop("ts", None)

        payload = None
        if payload_size > 0:
            payload = await reader.readexactly(payload_size)

        return cls(type=msg_type, metadata=header, payload=payload)


# ---------------------------------------------------------------------------
# Tensor serialization helpers
# ---------------------------------------------------------------------------

def serialize_tensor(tensor: mx.array) -> bytes:
    """Serialize an mx.array to bytes for network transfer."""
    # Use numpy as intermediate — fast and portable
    arr = np.array(tensor)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def deserialize_tensor(data: bytes) -> mx.array:
    """Deserialize bytes back to an mx.array."""
    buf = io.BytesIO(data)
    arr = np.load(buf)
    return mx.array(arr)


def serialize_hidden_states(hidden: mx.array, metadata: dict) -> Message:
    """Package hidden states for transfer between nodes."""
    return Message(
        type=MessageType.FORWARD,
        metadata={
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
            **metadata,
        },
        payload=serialize_tensor(hidden),
    )


def deserialize_hidden_states(msg: Message) -> mx.array:
    """Extract hidden states from a FORWARD or FORWARD_RESULT message."""
    return deserialize_tensor(msg.payload)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def make_join(cluster_secret: str, node_info: dict) -> Message:
    return Message(
        type=MessageType.JOIN,
        metadata={"secret": cluster_secret, "node": node_info},
    )


def make_join_ack(accepted: bool, capabilities: dict) -> Message:
    return Message(
        type=MessageType.JOIN_ACK,
        metadata={"accepted": accepted, "capabilities": capabilities},
    )


def make_load_layers(
    model_path: str,
    layer_start: int,
    layer_end: int,
    quantization: Optional[dict] = None,
) -> Message:
    return Message(
        type=MessageType.LOAD_LAYERS,
        metadata={
            "model_path": model_path,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "quantization": quantization or {},
        },
    )


def make_load_ack(
    success: bool,
    layers_loaded: int,
    memory_used_gb: float,
    memory_available_gb: float,
) -> Message:
    return Message(
        type=MessageType.LOAD_ACK,
        metadata={
            "success": success,
            "layers_loaded": layers_loaded,
            "memory_used_gb": memory_used_gb,
            "memory_available_gb": memory_available_gb,
        },
    )


def make_forward(
    hidden_states: mx.array,
    request_id: str,
    sequence_position: int,
    cache_id: Optional[str] = None,
) -> Message:
    return Message(
        type=MessageType.FORWARD,
        metadata={
            "request_id": request_id,
            "seq_pos": sequence_position,
            "cache_id": cache_id,
            "shape": list(hidden_states.shape),
            "dtype": str(hidden_states.dtype),
        },
        payload=serialize_tensor(hidden_states),
    )


def make_forward_result(
    hidden_states: mx.array,
    request_id: str,
    compute_time_ms: float,
) -> Message:
    return Message(
        type=MessageType.FORWARD_RESULT,
        metadata={
            "request_id": request_id,
            "compute_time_ms": compute_time_ms,
            "shape": list(hidden_states.shape),
            "dtype": str(hidden_states.dtype),
        },
        payload=serialize_tensor(hidden_states),
    )


def make_health() -> Message:
    return Message(type=MessageType.HEALTH, metadata={})


def make_health_ack(stats: dict) -> Message:
    return Message(type=MessageType.HEALTH_ACK, metadata=stats)


def make_error(code: str, message: str) -> Message:
    return Message(type=MessageType.ERROR, metadata={"code": code, "message": message})


def make_shutdown() -> Message:
    return Message(type=MessageType.SHUTDOWN, metadata={})


def make_cache_release(request_id: str) -> Message:
    """Tell a worker to drop the KV cache slot for a finished request."""
    return Message(
        type=MessageType.CACHE_OP,
        metadata={"op": "release", "request_id": request_id},
    )


def make_cache_ack(request_id: str, success: bool = True) -> Message:
    return Message(
        type=MessageType.CACHE_ACK,
        metadata={"request_id": request_id, "success": success},
    )
